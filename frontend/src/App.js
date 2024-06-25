import React, { useState } from 'react';
import './styles/styles.css';

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');

  const sendMessage = async () => {
    if (!input.trim()) return;

    const newMessage = { sender: 'user', content: input };
    setMessages([...messages, newMessage]);
    setInput('');

    try {
      const response = await fetch('http://localhost:8000/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: input })
      });
      const data = await response.json();
      const botMessage = { sender: 'bot', content: data.reply };
      setMessages([...messages, newMessage, botMessage]);
    } catch (error) {
      console.error('Error:', error);
    }
  };

  return (
    <div className="App">
      <div className="chat-container">
        <div className="chat-header">
          <div className="avatar"></div>
          <div className="chat-info">
            <div className="name">Jessica Cowles</div>
            <div className="status">We're online</div>
          </div>
        </div>
        <div className="chat-box">
          {messages.map((msg, index) => (
            <div key={index} className={`message ${msg.sender}`}>
              {msg.content}
            </div>
          ))}
        </div>
        <div className="chat-input">
          <div className="input-container">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Enter your message..."
            />
            <button onClick={sendMessage}>Send</button>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
