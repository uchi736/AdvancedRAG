import pandas as pd
from sqlalchemy import create_engine, text
from typing import List, Dict, Any, Optional, Tuple

class JargonDictionaryManager:
    """Manages the jargon dictionary in the database."""
    
    def __init__(self, connection_string: str, table_name: str = "jargon_dictionary"):
        self.connection_string = connection_string
        self.table_name = table_name
        self._init_jargon_table()
    
    def _init_jargon_table(self):
        """Initializes the jargon dictionary table and its indexes."""
        engine = create_engine(self.connection_string)
        with engine.connect() as conn:
            conn.execute(text(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id SERIAL PRIMARY KEY,
                    term TEXT UNIQUE NOT NULL,
                    definition TEXT NOT NULL,
                    domain TEXT,
                    aliases TEXT[],
                    related_terms TEXT[],
                    confidence_score FLOAT DEFAULT 1.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_jargon_term ON {self.table_name} (LOWER(term))"))
            conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_jargon_aliases ON {self.table_name} USING GIN(aliases)"))
            conn.commit()
    
    def add_term(self, term: str, definition: str, domain: Optional[str] = None,
                 aliases: Optional[List[str]] = None, related_terms: Optional[List[str]] = None,
                 confidence_score: float = 1.0) -> bool:
        """Adds or updates a term in the dictionary."""
        try:
            engine = create_engine(self.connection_string)
            with engine.connect() as conn:
                conn.execute(text(f"""
                    INSERT INTO {self.table_name} 
                    (term, definition, domain, aliases, related_terms, confidence_score)
                    VALUES (:term, :definition, :domain, :aliases, :related_terms, :confidence_score)
                    ON CONFLICT (term) DO UPDATE SET
                        definition = EXCLUDED.definition,
                        domain = EXCLUDED.domain,
                        aliases = EXCLUDED.aliases,
                        related_terms = EXCLUDED.related_terms,
                        confidence_score = EXCLUDED.confidence_score,
                        updated_at = CURRENT_TIMESTAMP
                """), {
                    "term": term, "definition": definition, "domain": domain,
                    "aliases": aliases or [], "related_terms": related_terms or [],
                    "confidence_score": confidence_score
                })
                conn.commit()
            return True
        except Exception as e:
            print(f"Error adding term to jargon dictionary: {e}")
            return False
    
    def lookup_terms(self, terms: List[str]) -> Dict[str, Dict[str, Any]]:
        """Looks up multiple terms from the dictionary."""
        if not terms:
            return {}
        
        engine = create_engine(self.connection_string)
        results = {}
        try:
            with engine.connect() as conn:
                placeholders = ', '.join([f':term_{i}' for i in range(len(terms))])
                query = text(f"""
                    SELECT term, definition, domain, aliases, related_terms, confidence_score
                    FROM {self.table_name}
                    WHERE LOWER(term) IN ({placeholders})
                    OR term = ANY(:aliases_check)
                """)
                params = {f"term_{i}": term.lower() for i, term in enumerate(terms)}
                params["aliases_check"] = terms
                
                rows = conn.execute(query, params).fetchall()
                for row in rows:
                    results[row.term] = {
                        "definition": row.definition, "domain": row.domain,
                        "aliases": row.aliases or [], "related_terms": row.related_terms or [],
                        "confidence_score": row.confidence_score
                    }
        except Exception as e:
            print(f"Error looking up terms: {e}")
        return results
    
    def bulk_import_from_csv(self, csv_path: str) -> Tuple[int, int]:
        """Bulk imports terms from a CSV file."""
        try:
            df = pd.read_csv(csv_path)
            success_count, error_count = 0, 0
            
            required = ["term", "definition"]
            if not all(col in df.columns for col in required):
                raise ValueError(f"CSV must contain columns: {required}")
            
            for _, row in df.iterrows():
                aliases = row.get("aliases", "").split("|") if "aliases" in row and pd.notna(row["aliases"]) else None
                related = row.get("related_terms", "").split("|") if "related_terms" in row and pd.notna(row["related_terms"]) else None
                
                if self.add_term(row["term"], row["definition"], row.get("domain"), aliases, related, row.get("confidence_score", 1.0)):
                    success_count += 1
                else:
                    error_count += 1
            return success_count, error_count
        except Exception as e:
            print(f"Error importing from CSV: {e}")
            return 0, -1
