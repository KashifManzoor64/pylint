from sentence_transformers import SentenceTransformer
import logging
import os
from odoo import models, api

_logger = logging.getLogger(__name__)


class MyChatbot(models.AbstractModel):
    _name = 'my.chatbot'
    _description = 'My Odoo Chatbot Logic'

    # Use a regular class variable instead of an Odoo model field for the embedding model
    _embedding_model = None  # This is a class-level variable, not bound to Odoo ORM

    @api.model
    def _get_embedding_model(self):
        # Check if the model is already loaded, if not, load it
        if MyChatbot._embedding_model is None:
            _logger.info("Initializing Sentence Transformer for embedding...")
            try:
                MyChatbot._embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception as e:
                _logger.error(f"Failed to load SentenceTransformer model: {e}")
                raise Exception("Embedding model could not be loaded. Check internet or installation.")
        return MyChatbot._embedding_model

    @api.model
    def get_chatbot_response(self, user_query):
        _logger.info(f"User query received: '{user_query}'")
        try:
            # Generate embedding for the user's query
            embedding_model = self._get_embedding_model()
            user_query_embedding = embedding_model.encode(user_query).tolist()
            _logger.info("User query embedding generated.")

            # Check if this is a count query and needs more comprehensive data
            is_count_query = any(word in user_query.lower() for word in ['how many', 'count', 'total', 'number of'])
            top_k = 10 if is_count_query else 3

            # Retrieve relevant context from your Odoo data
            relevant_chunks = self.env['chatbot.data.source'].retrieve_relevant_chunks(
                user_query_embedding,
                top_k=top_k,
                include_summary=is_count_query
            )

            if not relevant_chunks:
                _logger.info("No relevant context found in Odoo database.")
                return "I don't have enough specific information in my Odoo database to answer that question."

            context_text = "\n\n".join(relevant_chunks)
            _logger.info(f"Retrieved context:\n{context_text}")

            # Initialize Ollama LLM
            try:
                from langchain_community.llms import Ollama
                from langchain.prompts import PromptTemplate

                llm = Ollama(model="llama2", base_url="http://localhost:11434")
                _logger.info("Ollama LLM initialized.")
            except Exception as e:
                _logger.error(f"Failed to connect to Ollama server: {e}")
                return "I'm sorry, the AI assistant is currently unavailable. Please ensure Ollama server is running."

            # Generate the final response
            if is_count_query:
                prompt_template = PromptTemplate(
                    template="""You are a helpful Odoo ERP assistant. Answer the user's question about record counts based on the provided Odoo database information.

            The information includes model statistics and record details. When asked about counts, look for "Total records indexed" information for each model.

            Odoo Database Information:
            {context}

            User Query: {query}

            Answer the count question directly based on the indexed records shown above:""",
                    input_variables=["context", "query"]
                )
            else:
                prompt_template = PromptTemplate(
                    template="""You are a helpful Odoo ERP assistant. Your goal is to answer user questions about Odoo based **ONLY** on the provided "Odoo Database Information".
                            If the answer cannot be found in the provided information, respond with "I don't have enough specific information from the Odoo database to answer that question."
                            Odoo Database Information:
                            {context}
                            User Query: {query}
                            Answer:""",
                    input_variables=["context", "query"]
                )

            final_prompt = prompt_template.format(context=context_text, query=user_query)
            _logger.info(f"Sending prompt to Ollama LLM. Prompt length: {len(final_prompt)}")

            try:
                response = llm.invoke(final_prompt)
                _logger.info(f"Ollama LLM Response: {response}")
                return response
            except Exception as e:
                _logger.error(f"Error getting response from Ollama: {e}")
                return "An error occurred while getting a response from the AI."

        except Exception as e:
            _logger.error(f"Error in chatbot response generation: {e}")
            return f"I'm sorry, an error occurred while processing your request: {str(e)}"