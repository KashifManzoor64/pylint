from odoo import models, fields, api
import logging
import json
import numpy as np  # For cosine similarity calculation

_logger = logging.getLogger(__name__)

# Global flag to track if dependencies are available
HAS_SENTENCE_TRANSFORMERS = False


class ChatbotDataSource(models.Model):
    _name = 'chatbot.data.source'
    _description = 'Data Source for Chatbot'

    name = fields.Char(string="Record Name", required=True)
    odoo_model_id = fields.Reference(
        string="Source Record",
        selection='_get_reference_models',
        help="Reference to the original Odoo record."
    )
    content = fields.Text(string="Extracted Content", required=True,
                          help="The text chunk extracted from the Odoo record.")
    embedding = fields.Text(string="Vector Embedding",
                            help="Vector representation of the content, stored as a JSON string.")

    @api.model
    def _get_reference_models(self):
        """
        Get all models from the Lumina module and other relevant models
        to use as selection options for the reference field.
        """
        # Get the module ID for 'lumina'
        lumina_models = self.env['ir.model.data'].search([
            ('module', '=', 'lumina'),
            ('model', '=', 'ir.model')
        ])
        if lumina_models:
            model_ids = [model_data.res_id for model_data in lumina_models]
            module_models = self.env['ir.model'].browse(model_ids)
        else:
            module_models = self.env['ir.model'].search([('model', 'like', 'lumina.%')])

        # Add specific non-Lumina models
        additional_models = self.env['ir.model'].search([
            ('model', 'in', [
                'lumina.file',
                'lumina.calendar',
                'res.user'
            ])
        ])

        all_models = additional_models
        # all_models = module_models + additional_models
        return [(model.model, model.name) for model in all_models]

    @api.model
    def extract_and_prepare_data(self):
        """
        Extracts relevant data from various Odoo models and stores it in chatbot.data.source.
        This method now extracts data from all models in the Lumina module.
        """
        _logger.info("Starting data extraction for chatbot...")

        # Get all models from the Lumina module
        module_models = []

        # Get the module ID for 'lumina'
        lumina_models = self.env['ir.model.data'].sudo().search([
            ('module', '=', 'lumina'),
            ('model', '=', 'ir.model')
        ])
        if lumina_models:
            # Get all models that belong to the lumina module
            model_ids = [model_data.res_id for model_data in lumina_models]
            module_models = self.env['ir.model'].sudo().browse(model_ids)
            _logger.info(f"Found {len(module_models)} models in the Lumina module")
        else:
            _logger.warning("Lumina module not found")

        # Add specific non-Lumina models we want to include
















        additional_models = self.env['ir.model'].sudo().search([
            ('model', 'in', [
                'lumina.file',
                'lumina.calendar',
                'res.user'
            ])
        ])

        # Use all models now instead of just additional_models
        all_models = module_models + additional_models

        for model_info in all_models:
            model_name = model_info.model

            # Skip abstract models, transient models, and models without proper fields
            if model_name in self.env and not self.env[model_name]._abstract and not self.env[model_name]._transient:
                try:
                    # Start a new savepoint for each model
                    with self.env.cr.savepoint():
                        # Use sudo() to bypass access restrictions and disable record rules
                        records = self.env[model_name].sudo().with_context(active_test=False).search([])
                        _logger.info(f"Processing {len(records)} records from {model_name}")

                        # Get fields that might contain useful text data - use sudo()
                        fields_info = self.env[model_name].sudo().fields_get()
                        text_fields = [f for f, info in fields_info.items()
                                       if info['type'] in ('char', 'text', 'html') and not f.startswith('_')]

                        # Process each record
                        processed_count = 0
                        for record in records:
                            try:
                                # Use another savepoint for each record to handle individual failures
                                with self.env.cr.savepoint():
                                    # Use sudo() and disable record rules for individual record access
                                    record_sudo = record.sudo().with_context(active_test=False)

                                    # Skip records that don't have a name or display_name
                                    record_name = getattr(record_sudo, 'name',
                                                          getattr(record_sudo, 'display_name',
                                                                  f"Record {record_sudo.id}"))

                                    # Combine relevant fields into a single text chunk
                                    content_parts = [f"Model: {model_name}", f"Name: {record_name}"]

                                    for field in text_fields:
                                        try:
                                            field_value = getattr(record_sudo, field, None)
                                            if field_value and isinstance(field_value, str) and len(
                                                    field_value.strip()) > 0:
                                                content_parts.append(
                                                    f"{field.replace('_', ' ').title()}: {field_value}")
                                        except Exception as field_error:
                                            _logger.debug(
                                                f"Could not access field {field} for record {record_sudo.id} in {model_name}: {field_error}")
                                            continue

                                    combined_content = "\n".join(content_parts)

                                    # Only create records with meaningful content
                                    if len(combined_content) > 50:  # Minimum content length
                                        self.sudo().create({
                                            'name': record_name,
                                            'odoo_model_id': f'{model_name},{record_sudo.id}',
                                            'content': combined_content.strip(),
                                        })
                                        processed_count += 1

                            except Exception as record_error:
                                _logger.debug(f"Could not process record {record.id} in {model_name}: {record_error}")
                                continue

                        _logger.info(f"Successfully processed {processed_count} records from {model_name}")

                except Exception as e:
                    _logger.warning(f"Error extracting data from {model_name}: {e}")
                    # Continue with next model even if this one fails
                    continue

        _logger.info("All specified data extraction completed.")
        # Don't commit here, let the calling method handle it

    @api.model
    def generate_embeddings_for_data(self):
        """
        Generates embeddings for the extracted content in chatbot.data.source
        and stores them. This should be run after extract_and_prepare_data.
        """
        global HAS_SENTENCE_TRANSFORMERS

        # Try to import SentenceTransformer here instead of at module level
        try:
            from sentence_transformers import SentenceTransformer
            HAS_SENTENCE_TRANSFORMERS = True
        except ImportError as e:
            _logger.warning(f"Could not import sentence_transformers: {e}")
            return False

        _logger.info("Loading sentence transformer model for embeddings...")
        try:
            # This model will be downloaded the first time it's used (requires internet access)
            # 'all-MiniLM-L6-v2' is a good balance of size, speed, and quality for local use
            model = SentenceTransformer('all-MiniLM-L6-v2')
            _logger.info("Sentence Transformer model loaded successfully.")
        except Exception as e:
            _logger.error(f"Failed to load SentenceTransformer model: {e}")
            _logger.error(
                "Please ensure you have internet access and 'sentence-transformers' is installed: pip install sentence-transformers")
            return False

        # Find records that have content but no embedding yet
        records_to_process = self.search([('content', '!=', False), ('embedding', '=', False)])
        if not records_to_process:
            _logger.info("No new records to generate embeddings for.")
            return True

        _logger.info(f"Generating embeddings for {len(records_to_process)} records...")

        # Process in batches if you have a lot of records, to save memory/time
        batch_size = 50
        for i in range(0, len(records_to_process), batch_size):
            batch_records = records_to_process[i:i + batch_size]
            batch_contents = [rec.content for rec in batch_records]

            try:
                # Generate embeddings for the batch
                batch_vectors = model.encode(batch_contents).tolist()

                # Update each record with its embedding
                for j, record in enumerate(batch_records):
                    record.embedding = json.dumps(batch_vectors[j])  # Store as JSON string

                self.env.cr.commit()  # Commit batch changes
                _logger.info(f"Processed batch {i // batch_size + 1}. Total processed: {i + len(batch_records)}")
            except Exception as e:
                _logger.error(f"Error processing batch {i // batch_size + 1}: {e}")
                self.env.cr.rollback()  # Rollback current batch if error

        _logger.info("All specified embeddings generation completed.")
        return True

    @api.model
    def full_reindex_chatbot_data(self):
        """
        A helper function to run both extraction and embedding generation.
        You can call this from a Server Action or Cron Job.
        """
        _logger.info("Starting full re-indexing of chatbot data...")

        try:
            # Clear existing data before re-indexing
            with self.env.cr.savepoint():
                existing_records = self.search([])
                if existing_records:
                    existing_records.unlink()
                    _logger.info(f"Cleared {len(existing_records)} existing records")

            # Extract data
            with self.env.cr.savepoint():
                self.extract_and_prepare_data()
                _logger.info("Data extraction completed successfully")

            # Try to generate embeddings if the library is available
            global HAS_SENTENCE_TRANSFORMERS
            try:
                from sentence_transformers import SentenceTransformer
                HAS_SENTENCE_TRANSFORMERS = True
                with self.env.cr.savepoint():
                    self.generate_embeddings_for_data()
                    _logger.info("Embeddings generation completed successfully")
            except ImportError:
                _logger.warning("Skipping embedding generation due to missing dependencies")
            except Exception as e:
                _logger.error(f"Error during embedding generation: {e}")

            # Final commit
            self.env.cr.commit()
            _logger.info("Full re-indexing complete.")

        except Exception as e:
            _logger.error(f"Error during full reindex: {e}")
            self.env.cr.rollback()
            raise

    @api.model
    def retrieve_relevant_chunks(self, query_embedding, top_k=3, include_summary=False):
        """
        Retrieves the most similar data chunks from the database
        based on a query embedding.
        """
        all_data_sources = self.search([('embedding', '!=', False)])
        if not all_data_sources:
            _logger.warning("No data sources found for retrieval.")
            return []

        # For count/summary queries, include comprehensive model statistics
        if include_summary:
            model_stats = {}

            # Get ALL records, not just the ones we'll return for similarity
            for rec in all_data_sources:
                if rec.odoo_model_id:
                    try:
                        if hasattr(rec.odoo_model_id, '_name'):
                            model_name = rec.odoo_model_id._name
                        else:
                            model_name = str(rec.odoo_model_id).split(',')[0]

                        if model_name not in model_stats:
                            model_stats[model_name] = 0
                        model_stats[model_name] += 1
                    except Exception as e:
                        _logger.error(f"Error processing model reference for record {rec.id}: {e}")
                        continue

            # Also get actual counts from the source models for verification
            summary_chunks = []
            summary_chunks.append("=== DATABASE RECORD COUNTS ===")

            for model_name, indexed_count in model_stats.items():
                try:
                    # Get actual count from the source model
                    if model_name in self.env:
                        actual_count = self.env[model_name].search_count([])
                        summary_chunks.append(f"Model: {model_name}")
                        summary_chunks.append(f"  - Indexed in chatbot: {indexed_count} records")
                        summary_chunks.append(f"  - Total in database: {actual_count} records")
                    else:
                        summary_chunks.append(f"Model: {model_name} - Indexed: {indexed_count} records")
                except Exception as e:
                    summary_chunks.append(
                        f"Model: {model_name} - Indexed: {indexed_count} records (could not verify total)")
                    _logger.error(f"Error getting count for model {model_name}: {e}")

            summary_chunks.append("=== END COUNTS ===")

            # For count queries, return summary + ALL records, not just top_k
            if include_summary:
                all_content_chunks = []
                for rec in all_data_sources:
                    all_content_chunks.append(rec.content)

                return summary_chunks + all_content_chunks

        # For non-count queries, do similarity matching
        query_embedding_np = np.array(query_embedding)
        scored_chunks = []

        for rec in all_data_sources:
            try:
                data_embedding_np = np.array(json.loads(rec.embedding))
                similarity = np.dot(query_embedding_np, data_embedding_np) / (
                        np.linalg.norm(query_embedding_np) * np.linalg.norm(data_embedding_np))
                scored_chunks.append((similarity, rec.content))
            except Exception as e:
                _logger.error(f"Error calculating similarity for record {rec.id}: {e}")
                continue

        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        return [chunk[1] for chunk in scored_chunks[:top_k]]
