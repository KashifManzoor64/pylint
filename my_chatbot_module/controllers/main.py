from odoo import http
from odoo.http import request
import logging

_logger = logging.getLogger(__name__)

class MyChatbotController(http.Controller):
    @http.route('/my_chatbot/ask_ai', type='http', auth='user', methods=['POST'], website=True)
    def ask_ai_form(self, query=None, **kw):
        """
        HTTP endpoint for the chatbot form submission. Receives user query and returns a rendered template.
        """
        try:
            if not query:
                return request.render('my_chatbot_module.chatbot_assistant_template', {
                    'error_message': 'Please enter a question'
                })

            # Call the chatbot logic defined in my.chatbot model
            response = request.env['my.chatbot'].get_chatbot_response(query)

            # Render the template with the response
            return request.render('my_chatbot_module.chatbot_assistant_template', {
                'user_query': query,
                'ai_response': response
            })
        except Exception as e:
            _logger.error(f"Error in /my_chatbot/ask_ai controller: {e}")
            return request.render('my_chatbot_module.chatbot_assistant_template', {
                'error_message': f"An internal error occurred with the chatbot service: {str(e)}"
            })

    @http.route('/my_chatbot/assistant', type='http', auth='user', website=True)
    def chatbot_assistant(self, **kwargs):
        """
        Renders the chatbot assistant page
        """
        return request.render('my_chatbot_module.chatbot_assistant_template', {})
