import {
  FiMessageSquare,
  FiPlus,
  FiBookOpen,
  FiSettings,
} from "react-icons/fi";

interface Conversation {
  id: number;
  title: string;
}

interface SidebarProps {
  onNewChat: () => void;
  recentChats: Conversation[];
}

export default function Sidebar({
  onNewChat,
  recentChats,
}: SidebarProps) {
  return (
    <aside className="sidebar">
      {/* Logo */}

      <div className="sidebar-content">
        
        <div className="sidebar-header">
          <div className="logo">🇳🇵</div>

          <div className="logo-text">
            <h2>Nepal Budget AI</h2>
            <p>Annual Budget Assistant</p>
          </div>
        </div>

        {/* New Chat */}

        <button
          className="new-chat-btn"
          onClick={onNewChat}
        >
          <FiPlus />
          <span>नयाँ कुराकानी</span>
        </button>

        {/* Recent Chats */}

        <div className="sidebar-section">
          <h4>Recent Chats</h4>

          <ul>
            {recentChats.length === 0 ? (
              <li>
                <span>No conversations yet</span>
              </li>
            ) : (
              recentChats.map((chat) => (
                <li key={chat.id}>
                  <FiMessageSquare />

                  <span>
                    {chat.title.length > 30
                      ? chat.title.slice(0, 30) + "..."
                      : chat.title}
                  </span>
                </li>
              ))
            )}
          </ul>
        </div>

        {/* Resources */}

        <div className="sidebar-section">
          <h4>Resources</h4>

          <ul>
            <li>
              <FiBookOpen />
              <span>Budget Documents</span>
            </li>

            <li>
              <FiBookOpen />
              <span>Economic Survey</span>
            </li>
          </ul>
        </div>
      </div>

      {/* Footer */}

      <div className="sidebar-footer">
        <button className="sidebar-btn">
          <FiSettings />
          <span>Settings</span>
        </button>
      </div>
    </aside>
  );
}