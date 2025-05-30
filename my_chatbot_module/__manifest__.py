{
    'name': 'My Odoo AI Chatbot',
    'summary': 'AI Chatbot integrated with Odoo data using Ollama and pgvector',
    'version': '1.0',
    'category': 'Sales/Sales',
    'author': 'Your Name',
    'depends': ['base', 'mail', 'web', 'product'],
    'data': [
        'security/ir.model.access.csv', # You'll need to create this for your new models
        'views/livechat_extension.xml',
        'views/menu.xml', # You'll define views for ChatbotDataSource if you want to manage it from UI
        'data/server_actions.xml', # To run the indexing via Server Action
    ],
    'assets': {
    'web.assets_frontend': [
        '/my_chatbot_module/static/src/js/chatbot_livechat.js',
    ],
},
    'installable': True,
    'application': True,
    'auto_install': False,
    'license': 'LGPL-3',
}