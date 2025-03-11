import os
import pickle
import asyncio
import httpx
import xml.etree.ElementTree as ET
import numpy as np
from typing import List, Optional, NoReturn, Self, Tuple


class OpenAIModel:
    def __init__(self,
                 xml_folder: str | os.PathLike,
                 api_key: str,
                 model_name: str = 'gpt-3.5-turbo-0125',
                 embedding_model_name: str = 'text-embedding-3-small',
                 cache_file: str | os.PathLike = "vector_cache.pkl",
                 system_prompt: Optional[str] = None) -> Self:
        self.XML_FOLDER = xml_folder
        self.API_KEY = api_key
        self.MODEL_NAME = model_name
        self.EMBEDDING_MODEL_NAME = embedding_model_name
        self.CACHE_FILE = cache_file
        self.SYSTEM_PROMPT = system_prompt

    def _parse_xml_files(self) -> Tuple[List[str], List[str]] | NoReturn:
        texts = []
        files = []
        if not os.path.exists(self.XML_FOLDER):
            raise FileNotFoundError(f'The system cannot find the path specified: {self.XML_FOLDER}')
        for file in os.listdir(self.XML_FOLDER):
            if file.endswith(".xml"):
                file = os.path.join(self.XML_FOLDER, file)
                tree = ET.parse(file)
                root = tree.getroot()
                text_content = " ".join([elem.text for elem in root.iter() if elem.text])
                texts.append(text_content)
                files.append(file)
        if len(files) == 0:
            raise FileNotFoundError(f'Didnt find any .xml files in {self.XML_FOLDER} folder')
        return texts, files

    async def _get_openai_embedding(self, text: str) -> List[float] | NoReturn:
        url = "https://api.openai.com/v1/embeddings"
        headers = {"Authorization": f"Bearer {self.API_KEY}"}
        json_data = {"model": self.EMBEDDING_MODEL_NAME, "input": text}
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=json_data, headers=headers)
            response.raise_for_status()
            return response.json()["data"][0]["embedding"]

    async def load_and_vectorize_knowledge(self, revectorize: bool = False) -> str | List[str] | NoReturn:
        if os.path.exists(self.CACHE_FILE) and revectorize == False:
            with open(self.CACHE_FILE, "rb") as f:
                self.knowledge_base = pickle.load(f)
                return [self.CACHE_FILE]
        texts, files = self._parse_xml_files()
        vectors = await asyncio.gather(*(self._get_openai_embedding(text) for text in texts))
        knowledge_data = {"texts": texts, "vectors": vectors}
        with open(self.CACHE_FILE, "wb") as f:
            pickle.dump(knowledge_data, f)
        self.knowledge_base = knowledge_data
        return files

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        vec1, vec2 = np.array(vec1), np.array(vec2)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    async def _retrieve_relevant_knowledge(self, query: str, top_k: int = 3) -> List[str] | NoReturn:
        query_vector = await self._get_openai_embedding(query)
        similarities = [self._cosine_similarity(query_vector, kb_vector) for kb_vector in self.knowledge_base["vectors"]]
        top_indices = np.argsort(similarities)[-top_k:][::-1]  # Get top-k indices
        return [self.knowledge_base["texts"][i] for i in top_indices]

    async def generate_response(self,
                                query: str,
                                top_k: int = 3,
                                temperature: float = 0.7,
                                top_p: float = 0.9) -> str | NoReturn:
        relevant_knowledge = await self._retrieve_relevant_knowledge(query, top_k)
        context = "\n\n".join(relevant_knowledge)
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.API_KEY}"}
        json_data = {
            "model": self.MODEL_NAME,
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": f"{query}\n\nKnowledge Base:\n{context}"},
            ],
            "temperature": temperature,
            "top_p": top_p,
        }
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=json_data, headers=headers)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]

    async def check_openai_api(self) -> bool:
        url = "https://api.openai.com/v1/models"
        headers = {"Authorization": f"Bearer {self.API_KEY}"}
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(url, headers=headers)
            try:
                response.raise_for_status()
                return True
            except: return False