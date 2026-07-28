import { useState } from "react";

import Sidebar from "../components/Sidebar";
import Navbar from "../components/Navbar";
import Welcome from "../components/Welcome";
import PromptCards from "../components/PromptCards";
import ChatArea from "../components/ChatArea";
import ChatInput from "../components/ChatInput";

import "../styles/chat.css";

interface Message {
  id: number;
  sender: "user" | "assistant";
  text: string;
}

export default function Chat() {
  const [messages, setMessages] = useState<Message[]>([]);

  const handleSendMessage = (message: string) => {
    if (!message.trim()) return;

    setMessages((prev) => [
      ...prev,
      {
        id: Date.now(),
        sender: "user",
        text: message,
      },
    ]);

    // Backend API यहाँ call हुनेछ
  };

  return (
    <div className="chat-layout">
      <Sidebar />

      <main className="main-content">
        <Navbar />

        <div className="chat-body">
          {messages.length === 0 ? (
            <>
              <Welcome />

              <PromptCards />
            </>
          ) : (
            <ChatArea messages={messages} />
          )}
        </div>

        <ChatInput
          onSendMessage={handleSendMessage}
        />
      </main>
    </div>
  );
}