import arxiv
import json
import os
import time
from typing import Dict
import google.generativeai as genai
from google.api_core import exceptions
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import streamlit as st

# Configuration
PAPER_DIR = "papers"
GOOGLE_API_KEY = "AIzaSyAxwloFIqGSiYe-1EdhPT_O1CvJwel2GIs"  # Replace with your actual API key

# Initialize Gemini AI
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")

# Ensure papers directory exists
os.makedirs(PAPER_DIR, exist_ok=True)

class ResearchApp:
    def __init__(self):
        self._setup_ui()
        
    def _load_papers_info(self, file_path: str) -> Dict:
        """Helper method to load papers info from JSON file"""
        try:
            with open(file_path, "r") as json_file:
                return json.load(json_file)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def search_papers(self, topic: str, max_results: int = 5) -> Dict:
        """Search for papers on arXiv and store their metadata."""
        client = arxiv.Client()
        search = arxiv.Search(
            query=topic,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )

        # Create topic-specific directory
        topic_dir = os.path.join(PAPER_DIR, topic.lower().replace(" ", "_"))
        os.makedirs(topic_dir, exist_ok=True)
        file_path = os.path.join(topic_dir, "papers_info.json")

        # Load existing data or initialize new
        papers_info = self._load_papers_info(file_path)

        # Process new papers
        paper_ids = []
        for paper in client.results(search):
            paper_id = paper.get_short_id()
            paper_ids.append(paper_id)

            papers_info[paper_id] = {
                'title': paper.title,
                'authors': [author.name for author in paper.authors],
                'summary': paper.summary,
                'pdf_url': paper.pdf_url,
                'published': str(paper.published.date()),
                'abs_url': paper.pdf_url.replace('pdf', 'abs')
            }

        # Save updated data
        with open(file_path, "w") as json_file:
            json.dump(papers_info, json_file, indent=2)

        return {
            "paper_ids": paper_ids,
            "save_path": file_path,
            "message": f"Found {len(paper_ids)} papers on '{topic}'"
        }

    def get_paper_content(self, paper_id: str) -> Dict:
        """Get detailed content for a specific paper directly from arXiv."""
        try:
            paper = next(arxiv.Client().results(arxiv.Search(id_list=[paper_id.split()[0]])))
            return {
                "title": paper.title,
                "abstract": paper.summary,
                "pdf_url": paper.pdf_url,
                "abs_url": paper.pdf_url.replace('pdf', 'abs'),
                "authors": [author.name for author in paper.authors],
                "published": str(paper.published.date())
            }
        except Exception as e:
            return {"error": f"Failed to retrieve paper {paper_id}: {str(e)}"}

    def extract_info(self, paper_id: str) -> Dict:
        """Search for information about a paper in local storage."""
        for item in os.listdir(PAPER_DIR):
            item_path = os.path.join(PAPER_DIR, item)
            if os.path.isdir(item_path):
                file_path = os.path.join(item_path, "papers_info.json")
                if os.path.isfile(file_path):
                    try:
                        with open(file_path, "r") as json_file:
                            papers_info = json.load(json_file)
                            if paper_id in papers_info:
                                return papers_info[paper_id]
                    except (FileNotFoundError, json.JSONDecodeError) as e:
                        continue

        return {"error": f"No local information found for paper {paper_id}"}

    def research_assistant(self, query: str) -> Dict:
        """Use Gemini AI to answer research-related questions."""
        try:
            response = model.generate_content(
                f"User bertanya tentang proposal {query}. respon seperti asisten pencarian web. dan jelaskan tentang {query}",
                safety_settings={
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                }
            )
            return {"response": response.text}
        except exceptions.TooManyRequests:
            time.sleep(2)
            return {"error": "Too many requests, please wait"}
        except Exception as e:
            return {"error": str(e)}

    def _display_paper_info(self, paper_info: Dict):
        """Display paper information in Streamlit format"""
        st.subheader(paper_info['title'])
        st.write(f"**Authors:** {', '.join(paper_info['authors'])}")
        st.write(f"**Published:** {paper_info['published']}")
        st.markdown("---")
        st.subheader("Abstract")
        st.write(paper_info['summary'])
        st.markdown(f"[Download PDF]({paper_info['pdf_url']})", unsafe_allow_html=True)

    def _setup_ui(self):
        """Set up the Streamlit user interface"""
        st.set_page_config(page_title="Research Assistant", layout="wide")
        st.title("Research Assistant Chatbot")
        
        # Main input area
        user_input = st.text_input(
            "Ask a research question or enter a paper ID (e.g., 2401.12345):",
            placeholder="Your question or paper ID..."
        )
        
        if st.button("Submit") and user_input:
            with st.spinner("Processing your request..."):
                # Check if input looks like a paper ID
                if any(c.isdigit() for c in user_input):
                    # Try to get paper info from local storage first
                    paper_info = self.extract_info(user_input)
                    if 'error' in paper_info:
                        # If not found locally, try to fetch from arXiv
                        paper_info = self.get_paper_content(user_input)
                    
                    if 'error' not in paper_info:
                        self._display_paper_info(paper_info)
                        return
                
                # Default research assistant response
                response = self.research_assistant(user_input)
                result = response.get('response', response.get('error', 'No response available'))
                
                st.subheader("Research Assistant Response")
                st.markdown(result)
        
        # API functions section
        st.markdown("---")
        st.subheader("API Functions")
        
        with st.expander("Search Papers"):
            topic = st.text_input("Research topic:")
            max_results = st.number_input("Max results:", min_value=1, max_value=20, value=5)
            if st.button("Search"):
                with st.spinner("Searching papers..."):
                    result = self.search_papers(topic, max_results)
                    st.json(result)
        
        with st.expander("Get Paper Content"):
            paper_id = st.text_input("Paper ID:")
            if st.button("Get Content"):
                with st.spinner("Fetching paper..."):
                    result = self.get_paper_content(paper_id)
                    st.json(result)
        
        with st.expander("Research Assistant API"):
            query = st.text_input("Research question:")
            if st.button("Ask Assistant"):
                with st.spinner("Generating response..."):
                    result = self.research_assistant(query)
                    st.json(result)

if __name__ == "__main__":
    app = ResearchApp()
