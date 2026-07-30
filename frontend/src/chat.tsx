import { useState } from "react";
import axios from "axios";

import Sidebar from "./components/Sidebar";
import Navbar from "./components/Navbar";
import Welcome from "./components/Welcome";
import PromptCards from "./components/PromptCards";
import ChatArea from "./components/ChatArea";
import ChatInput from "./components/ChatInput";

import "../styles/chat.css";

interface Message {
  id: number;
  sender: "user" | "assistant";
  text: string;
}

export default function Chat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);

  const handleSendMessage = async (question: string) => {
    // User को message तुरुन्त UI मा देखाउने
    const userMessage: Message = {
      id: Date.now(),
      sender: "user",
      text: question,
    };

    setMessages((prev) => [...prev, userMessage]);

    setLoading(true);

    try {
      // Login गर्दा save भएको JWT निकाल्ने
      const token = localStorage.getItem("token");

      console.log("Before axios");

      // Backend call
      const response = await axios.post(
        "http://127.0.0.1:8000/chat",
        {
          question: question,
        },
        {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        }
      );

      console.log("After axios");
      
      // Backend बाट आएको उत्तर
      const botMessage: Message = {
        id: Date.now() + 1,
        sender: "assistant",
        text: response.data.answer,
      };

      setMessages((prev) => [...prev, botMessage]);
    } catch (error) {
      console.error(error);

      setMessages((prev) => [
        ...prev,
        {
          id: Date.now() + 1,
          sender: "assistant",
          text: "उत्तर प्राप्त गर्न समस्या भयो।",
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="chat-layout">
      <Sidebar />

      <main className="main-content">
        <Navbar />

        <Welcome />

        <PromptCards />

        <ChatArea messages={messages} />

        {loading && (
          <p style={{ margin: "10px" }}>
            सोच्दैछु...
          </p>
        )}

        <ChatInput
          onSendMessage={handleSendMessage}
        />
      </main>
    </div>
  );
}