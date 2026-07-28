import { useState } from "react";
import { FiSend } from "react-icons/fi";

interface ChatInputProps {
  onSendMessage?: (message: string) => void;
}

export default function ChatInput({
  onSendMessage,
}: ChatInputProps) {
  const [message, setMessage] = useState("");

  const handleSubmit = (
    e: React.FormEvent<HTMLFormElement>
  ) => {
    e.preventDefault();

    const trimmedMessage = message.trim();

    if (!trimmedMessage) return;

    onSendMessage?.(trimmedMessage);

    setMessage("");
  };

  return (
    <form
      className="chat-input-container"
      onSubmit={handleSubmit}
    >
      <input
        type="text"
        placeholder="नेपालको वार्षिक बजेट सम्बन्धी प्रश्न सोध्नुहोस्..."
        value={message}
        onChange={(e) =>
          setMessage(e.target.value)
        }
      />

      <button type="submit">
        <FiSend size={20} />
      </button>
    </form>
  );
}