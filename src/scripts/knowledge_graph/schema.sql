-- Knowledge Graph Database Schema
-- Required: PostgreSQL with pgvector extension
-- Author: Advanced RAG System
-- Version: 2.0

-- ============================================
-- 1. Extensions
-- ============================================
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================
-- 2. Node Table
-- ============================================
DROP TABLE IF EXISTS knowledge_edges CASCADE;
DROP TABLE IF EXISTS knowledge_nodes CASCADE;

CREATE TABLE knowledge_nodes (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    node_type VARCHAR(50) NOT NULL CHECK (
        node_type IN ('Term', 'Category', 'Domain', 'Component', 'System')
    ),
    term VARCHAR(255),
    definition TEXT,
    properties JSONB DEFAULT '{}',
    embedding vector(1536),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Partial unique index for Term nodes only
CREATE UNIQUE INDEX uniq_term_only_term_nodes
ON knowledge_nodes (term)
WHERE node_type = 'Term';

-- HNSW index for vector similarity search
CREATE INDEX idx_nodes_embedding_hnsw
ON knowledge_nodes
USING hnsw (embedding vector_l2_ops)
WITH (m = 16, ef_construction = 64);

-- Other indexes
CREATE INDEX idx_nodes_type ON knowledge_nodes(node_type);
CREATE INDEX idx_nodes_properties ON knowledge_nodes USING GIN(properties);

-- ============================================
-- 3. Edge Table
-- ============================================
CREATE TABLE knowledge_edges (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_id UUID REFERENCES knowledge_nodes(id) ON DELETE CASCADE,
    target_id UUID REFERENCES knowledge_nodes(id) ON DELETE CASCADE,
    edge_type VARCHAR(50) NOT NULL CHECK (
        edge_type IN (
            -- Hierarchical relations
            'IS_A',           -- 〜の一種である (specific→general)
            'HAS_SUBTYPE',    -- 下位分類を持つ (general→specific)
            'BELONGS_TO',     -- カテゴリに属する (term→category)
            
            -- Compositional relations
            'PART_OF',        -- 〜の一部である (part→whole)
            'HAS_COMPONENT',  -- 構成要素を持つ (whole→part)
            'INCLUDES',       -- 含む/包含する (container→contained)
            
            -- Functional relations
            'USED_FOR',       -- 〜に使用される (tool→purpose)
            'PERFORMS',       -- 〜を実行する (agent→action)
            'CONTROLS',       -- 〜を制御する (controller→controlled)
            'MEASURES',       -- 〜を測定する (instrument→measured)
            
            -- Association relations
            'RELATED_TO',     -- 関連する (bidirectional)
            'SIMILAR_TO',     -- 類似する (bidirectional)
            'SYNONYM',        -- 同義語 (bidirectional)
            'CO_OCCURS_WITH', -- 共起する (bidirectional)
            'DEPENDS_ON',     -- 依存する (dependent→dependency)
            
            -- Process relations
            'CAUSES',         -- 引き起こす (cause→effect)
            'PREVENTS',       -- 防止する (preventer→prevented)
            'PROCESSES',      -- 処理する (processor→processed)
            'GENERATES'       -- 生成する (generator→generated)
        )
    ),
    weight FLOAT DEFAULT 1.0 CHECK (weight >= 0 AND weight <= 1),
    confidence FLOAT DEFAULT 1.0 CHECK (confidence >= 0 AND confidence <= 1),
    properties JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT no_self_loop CHECK (source_id != target_id),
    CONSTRAINT unique_edge UNIQUE(source_id, target_id, edge_type)
);

-- Edge indexes
CREATE INDEX idx_edges_source ON knowledge_edges(source_id);
CREATE INDEX idx_edges_target ON knowledge_edges(target_id);
CREATE INDEX idx_edges_type ON knowledge_edges(edge_type);
CREATE INDEX idx_edges_weight ON knowledge_edges(weight DESC);
CREATE INDEX idx_edges_confidence ON knowledge_edges(confidence DESC);
CREATE INDEX idx_edges_properties ON knowledge_edges USING GIN(properties);

-- ============================================
-- 4. Triggers
-- ============================================

-- Auto-update updated_at timestamp
CREATE OR REPLACE FUNCTION set_updated_at()
RETURNS trigger AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_nodes_updated
BEFORE UPDATE ON knowledge_nodes
FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- ============================================
-- 5. Helper Functions
-- ============================================

-- Get node by term
CREATE OR REPLACE FUNCTION get_node_by_term(p_term VARCHAR)
RETURNS TABLE(
    id UUID,
    node_type VARCHAR,
    term VARCHAR,
    definition TEXT,
    properties JSONB,
    embedding vector,
    created_at TIMESTAMP,
    updated_at TIMESTAMP
) AS $$
BEGIN
    RETURN QUERY
    SELECT n.*
    FROM knowledge_nodes n
    WHERE n.term = p_term
    LIMIT 1;
END;
$$ LANGUAGE plpgsql;

-- Get edges for a node
CREATE OR REPLACE FUNCTION get_edges_for_node(p_node_id UUID)
RETURNS TABLE(
    edge_id UUID,
    source_id UUID,
    target_id UUID,
    edge_type VARCHAR,
    weight FLOAT,
    confidence FLOAT,
    properties JSONB,
    direction VARCHAR
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        e.id as edge_id,
        e.source_id,
        e.target_id,
        e.edge_type,
        e.weight,
        e.confidence,
        e.properties,
        'outgoing'::VARCHAR as direction
    FROM knowledge_edges e
    WHERE e.source_id = p_node_id
    
    UNION ALL
    
    SELECT 
        e.id as edge_id,
        e.source_id,
        e.target_id,
        e.edge_type,
        e.weight,
        e.confidence,
        e.properties,
        'incoming'::VARCHAR as direction
    FROM knowledge_edges e
    WHERE e.target_id = p_node_id;
END;
$$ LANGUAGE plpgsql;

-- Get subgraph around a node (BFS)
CREATE OR REPLACE FUNCTION get_subgraph(
    p_center_id UUID,
    p_max_depth INTEGER DEFAULT 2
)
RETURNS TABLE(
    node_id UUID,
    depth INTEGER
) AS $$
DECLARE
    current_depth INTEGER := 0;
BEGIN
    -- Create temp table for BFS
    CREATE TEMP TABLE IF NOT EXISTS bfs_nodes (
        node_id UUID PRIMARY KEY,
        depth INTEGER
    ) ON COMMIT DROP;
    
    -- Start with center node
    INSERT INTO bfs_nodes VALUES (p_center_id, 0);
    
    -- BFS traversal
    WHILE current_depth < p_max_depth LOOP
        INSERT INTO bfs_nodes
        SELECT DISTINCT 
            CASE 
                WHEN e.source_id IN (SELECT node_id FROM bfs_nodes WHERE depth = current_depth) 
                THEN e.target_id 
                ELSE e.source_id 
            END,
            current_depth + 1
        FROM knowledge_edges e
        WHERE (
            e.source_id IN (SELECT node_id FROM bfs_nodes WHERE depth = current_depth)
            OR e.target_id IN (SELECT node_id FROM bfs_nodes WHERE depth = current_depth)
        )
        AND NOT EXISTS (
            SELECT 1 FROM bfs_nodes bn 
            WHERE bn.node_id = CASE 
                WHEN e.source_id IN (SELECT node_id FROM bfs_nodes WHERE depth = current_depth) 
                THEN e.target_id 
                ELSE e.source_id 
            END
        );
        
        current_depth := current_depth + 1;
    END LOOP;
    
    RETURN QUERY SELECT * FROM bfs_nodes;
END;
$$ LANGUAGE plpgsql;

-- ============================================
-- 6. Views
-- ============================================

-- View for term relationships
CREATE OR REPLACE VIEW v_term_relationships AS
SELECT 
    n1.term as source_term,
    e.edge_type,
    n2.term as target_term,
    e.weight,
    e.confidence,
    e.properties->>'provenance' as provenance
FROM knowledge_edges e
JOIN knowledge_nodes n1 ON e.source_id = n1.id
JOIN knowledge_nodes n2 ON e.target_id = n2.id
WHERE n1.node_type = 'Term' AND n2.node_type = 'Term';

-- View for graph statistics
CREATE OR REPLACE VIEW v_graph_statistics AS
SELECT 
    (SELECT COUNT(*) FROM knowledge_nodes) as total_nodes,
    (SELECT COUNT(*) FROM knowledge_nodes WHERE node_type = 'Term') as term_nodes,
    (SELECT COUNT(*) FROM knowledge_nodes WHERE node_type = 'Category') as category_nodes,
    (SELECT COUNT(*) FROM knowledge_edges) as total_edges,
    (SELECT COUNT(DISTINCT edge_type) FROM knowledge_edges) as edge_types,
    (SELECT AVG(weight) FROM knowledge_edges) as avg_edge_weight,
    (SELECT AVG(confidence) FROM knowledge_edges) as avg_edge_confidence;

-- ============================================
-- 7. Sample Data (Optional)
-- ============================================

-- Uncomment to insert sample nodes
/*
INSERT INTO knowledge_nodes (node_type, term, definition, properties) VALUES
    ('Term', 'ピストン', 'シリンダ内で往復運動する部品', '{"c_value": 25.5, "frequency": 45}'),
    ('Term', 'エンジン', '動力を生成する機械', '{"c_value": 85.2, "frequency": 120}'),
    ('Category', 'エンジン部品', 'エンジンを構成する部品のカテゴリ', '{"cluster_id": 3}');

-- Sample edges
INSERT INTO knowledge_edges (source_id, target_id, edge_type, weight, confidence, properties)
SELECT 
    n1.id, n2.id, 'PART_OF', 0.95, 0.9, '{"provenance": "definition"}'
FROM knowledge_nodes n1, knowledge_nodes n2
WHERE n1.term = 'ピストン' AND n2.term = 'エンジン';
*/

-- ============================================
-- 8. Maintenance
-- ============================================

-- Analyze tables for query optimization
ANALYZE knowledge_nodes;
ANALYZE knowledge_edges;