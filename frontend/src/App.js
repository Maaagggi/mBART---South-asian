import React, { useState } from "react";

import { footer, welcomeText } from "./utils/constants";

import "./App.css";

function App() {
  // States below
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");

  const sendUserMessage = () => {
    if (input.trim()) {
      setMessages([...messages, { text: input, sender: "user" }]);
      setInput("");

      setTimeout(() => {
        setMessages((prevMessages) => [
          ...prevMessages,
          // Dummy text response from the bot when we type something
          { text: "This is a response from the chatbot.", sender: "bot" },
        ]);
      }, 1000);
    }
  };

  const handleKeyPress = (event) => {
    if (event.key === "Enter") {
      return sendUserMessage();
    }
  };

  return (
    <div>
      <h1 className="welcome-txt">Welcome to CaliBot</h1>
      <div className="chat-container">
        <div className="chat-interface">
          {messages.map((message, index) => (
            <div
              key={index}
              className={`message ${
                message.sender === "user" ? "user" : "bot"
              }`}
            >
              {message.text}
            </div>
          ))}
        </div>
        <div className="input-container">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder={welcomeText}
          />
          <button onClick={() => sendUserMessage()}>{footer.buttonName}</button>
        </div>
      </div>
    </div>
  );
}

export default App;
