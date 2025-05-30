odoo.define('my_chatbot_module.assistant', function (require) {
    'use strict';

    var ajax = require('web.ajax');
    var $ = require('jquery');  // Ensure jQuery is loaded
    deubgger
    $(document).ready(function() {
        var $chatMessages = $('#chat-messages');
        var $userInput = $('#user-input');
        var $sendButton = $('#send-button');
        var $exampleQuestions = $('.example-question');

        function addMessage(message, isUser) {
            var $message = $('<div class="p-3 mb-3 rounded message"></div>');
            if (isUser) {
                $message.addClass('bg-primary text-white ml-4');
                $message.prepend('<i class="fa fa-user-circle mr-2"></i>');
            } else {
                $message.addClass('bg-white border mr-4');
                $message.prepend('<i class="fa fa-robot mr-2"></i>');
            }

            var formattedMessage = message.replace(/\n/g, '<br>');
            $message.html($message.html() + formattedMessage);

            $chatMessages.append($message);
            $chatMessages.scrollTop($chatMessages[0].scrollHeight);
        }

        function sendMessage(message) {
            if (!message) {
                message = $userInput.val().trim();
            }

            if (message) {
                addMessage(message, true);
                $userInput.val('');
                $userInput.prop('disabled', true);
                $sendButton.prop('disabled', true);
                $sendButton.html('<i class="fa fa-spinner fa-spin"></i> Processing...');

                // Send to backend
                ajax.jsonRpc('/my_chatbot/ask_ai', 'call', {
                    'query': message
                }).then(function(response) {
                    if (response.error) {
                        addMessage(response.message, false);
                    } else {
                        addMessage(response, false);
                    }
                }).fail(function(error) {
                    addMessage('Sorry, an error occurred. Please try again.', false);
                    console.error(error);
                }).always(function() {
                    $userInput.prop('disabled', false);
                    $sendButton.prop('disabled', false);
                    $sendButton.html('<i class="fa fa-paper-plane mr-1"></i> Send');
                    $userInput.focus();
                });
            }
        }

        $sendButton.click(function() {
            sendMessage();
        });

        $userInput.keypress(function(e) {
            if (e.which === 13) {
                sendMessage();
                e.preventDefault();
            }
        });

        // Example questions click handler
        $exampleQuestions.click(function() {
            var question = $(this).data('question');
            $userInput.val(question);
            sendMessage(question);
        });

        // Add welcome message
        addMessage('Hello! I\'m your AI assistant for Lumina. How can I help you today?', false);
        $userInput.focus();
    });
});
