import unittest
import os
from tools.find_weather import get_weather
from graph_agent import lookup_policy
from dotenv import load_dotenv

load_dotenv()

class TestSystemIntegration(unittest.TestCase):

    def test_real_weather_api(self):
        """
        HITS REAL API: Checks if OpenWeatherMap is reachable and returning data.
        """
        city = "Hyderabad"
        print(f"\nTesting Weather API for {city}...")
        
        result = get_weather.invoke(city)
        
        print(f"   Response: {result}")
        
        # Verification
        self.assertIsInstance(result, str)
        self.assertNotIn("Error", result, "API returned an error message")
        self.assertIn("temperature", result.lower())
        print("Weather API: SUCCESS")

    def test_real_rag_retrieval(self):
        """
        HITS REAL DB: Checks if Qdrant can retrieve the 'one-shot' info you indexed.
        """
        query = "Explain one-shot prompting"
        print(f"\nTesting Real Qdrant Retrieval for: '{query}'...")
        
        try:
            result = lookup_policy.invoke(query)
            print(f"   Response Snippet: {result[:100]}...")
            self.assertTrue(len(result) > 50, "Retrieved content is suspiciously short")
            self.assertTrue(
                any(word in result.lower() for word in ["prompt", "example", "model", "shot"]),
                "Did not find expected keywords in retrieved text"
            )
            print("Real RAG Retrieval: SUCCESS")
            
        except Exception as e:
            self.fail(f"RAG tool crashed: {e}")

if __name__ == '__main__':
    print("STARTING INTEGRATION TESTS (REAL API CALLS)")
    unittest.main()
