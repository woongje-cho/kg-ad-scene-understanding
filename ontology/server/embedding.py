"""
Embedding Manager for generating and managing vector embeddings using OpenAI API.
"""
import os
from typing import List, Optional
from openai import OpenAI
import traceback
from openai import OpenAIError  # 파일 상단에 이미 OpenAI import 있으니, 이 줄 추가해도 됨
from openai import APIConnectionError  # 👈 추가
import time



class EmbeddingManager:
    """Manages embedding generation using OpenAI's text-embedding models."""

    def __init__(self, api_key: Optional[str] = None, model: str = "text-embedding-3-small", dimensions: int = 512):
        """
        Initialize EmbeddingManager with OpenAI API.

        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env variable)
            model: OpenAI embedding model name
                - 'text-embedding-3-small': 1536 dimensions (default), cost-effective
                - 'text-embedding-3-large': 3072 dimensions, higher performance
                - 'text-embedding-ada-002': 1536 dimensions, legacy model
            dimensions: Output vector dimensions (default: 512 for faster processing)
                - Lower dimensions = faster & less storage
                - Recommended: 512 (balanced), 256 (fast), 1536 (full quality)
        """
        #이게 문제였음!!!!!!!!!!!!!
        raw_key = api_key or os.getenv("OPENAI_API_KEY") or ""
        self.api_key = raw_key.strip()  # 앞뒤 공백 + \r\n 다 제거
        
        if not self.api_key:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable.")

        self.model = model
        self.dimensions = dimensions
        self.client = OpenAI(api_key=self.api_key)

        print(f"✓ EmbeddingManager initialized with model: {self.model}, dimensions: {self.dimensions}")


    def generate_embedding(self, text: str) -> List[float]:
        """Generate a single embedding with simple retry logic."""
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        for attempt in range(3):
            # 시도 번호에 따라 대기 시간 설정 (1초, 2초, 3초)
            wait = attempt + 1

            try:
                response = self.client.embeddings.create(
                    input=text,
                    model=self.model,
                    dimensions=self.dimensions
                )
                return response.data[0].embedding

            except APIConnectionError as e:
                # 네트워크 계열 에러 → 재시도
                print(
                    f"🔥 [Embedding] APIConnectionError "
                    f"(attempt {attempt+1}/3, wait {wait}s): {repr(e)}"
                )
                if e.__cause__:
                    print("   └ underlying cause:", repr(e.__cause__))
                time.sleep(wait)

            except Exception as e:
                # 다른 에러는 바로 위로 올려서 어떤 문제인지 보자
                print("🔥 [Embedding] Unexpected error!")
                print("   type:", type(e))
                print("   repr:", repr(e))
                traceback.print_exc()
                raise

        # 여기까지 왔다는 건 3번 다 실패했다는 뜻
        raise RuntimeError("Failed to generate embedding after 3 attempts")

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple text strings in batch.

        Args:
            texts: List of input texts to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        # Filter out empty texts
        valid_texts = [t for t in texts if t and t.strip()]
        if not valid_texts:
            raise ValueError("All texts are empty")

        response = self.client.embeddings.create(
            input=valid_texts,
            model=self.model,
            dimensions=self.dimensions
        )

        return [data.embedding for data in response.data]

    def embed_individual(self, individual, neo4j_session) -> bool:
        """
        Generate embedding for an OWL individual and store it in Neo4j.

        Args:
            individual: Owlready2 Individual object
            neo4j_session: Neo4j session for database operations

        Returns:
            True if embedding was created, False if skipped (no description)
        """
        # Get description from rdfs:comment
        description = None
        if hasattr(individual, 'comment') and individual.comment:
            description = individual.comment[0] if isinstance(individual.comment, list) else individual.comment

        if not description:
            return False

        # Generate embedding
        embedding = self.generate_embedding(description)

        # Store in Neo4j
        neo4j_session.run("""
            MATCH (n:Individual {id: $id})
            SET n.embedding = $embedding,
                n.description = $description
        """, id=individual.name, embedding=embedding, description=description)

        return True

    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings for the current configuration."""
        return self.dimensions
