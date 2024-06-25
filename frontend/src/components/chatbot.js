// components/Chatbot.js
import React, { useState } from 'react';
import './styles/styles.css'; // Import the main styles

const Chatbot = () => {
  const [messages, setMessages] = useState([]);

  const sendMessage = async () => {
    const userInput = document.getElementById('user-input').value.trim();
    if (!userInput) return;

    const userMessage = { id: messages.length + 1, sender: 'user', content: userInput };
    setMessages([...messages, userMessage]);

    try {
      const response = await fetch('http://localhost:8000/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ message: userInput })
      });

      const data = await response.json();
      const botMessage = { id: messages.length + 2, sender: 'bot', content: data.reply };
      setMessages([...messages, botMessage]);

    } catch (error) {
      console.error('Error:', error);
    }

    document.getElementById('user-input').value = '';
  };

  return (
    <section id="chat-interface">
      <div id="chat-box">
        {messages.map(message => (
          <div key={message.id} className={`message ${message.sender}`}>
            <span>{message.sender === 'bot' ? 'Bot:' : 'You:'}</span>
            {message.content}
          </div>
        ))}
      </div>
      <div id="disclaimer">
        <p>Chatbot can make mistakes. Check important info.</p>
      </div>
      <input type="text" id="user-input" placeholder="Type your message..." />
      <button onClick={sendMessage}>Send</button>
    </section>
  );
};

export default Chatbot;
