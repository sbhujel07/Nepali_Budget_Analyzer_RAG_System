interface Message {
  id: number;
  sender: "user" | "assistant";
  text: string;
}

interface ChatAreaProps {
  messages?: Message[];
}

export default function ChatArea({
  messages = [],
}: ChatAreaProps) {
  return (
    <section className="chat-area">

      {messages.length === 0 ? (
        <div className="empty-chat">

          <h2>कुराकानी सुरु गर्नुहोस्</h2>

          <p>
            नेपालको वार्षिक बजेट सम्बन्धी कुनै पनि प्रश्न तलको
            इनपुट बक्समा लेख्नुहोस्।
          </p>

        </div>
      ) : (
        <div className="messages">

          {messages.map((message) => (
            <div
              key={message.id}
              className={`message ${message.sender}`}
            >
              <p>{message.text}</p>
            </div>
          ))}

        </div>
      )}

    </section>
  );
}