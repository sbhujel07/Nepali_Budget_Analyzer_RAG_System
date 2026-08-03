import {
  FiMessageSquare,
  FiPlus,
  FiBookOpen,
  FiSettings,
  FiLogOut,
} from "react-icons/fi";

interface SidebarProps {
  onNewChat: () => void;
}

export default function Sidebar({
  onNewChat,
}: SidebarProps) {
  return (
    <aside className="sidebar">
      {/* Logo */}

      <div>
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
            <li>
              <FiMessageSquare />
              <span>शिक्षा बजेट</span>
            </li>

            <li>
              <FiMessageSquare />
              <span>स्वास्थ्य बजेट</span>
            </li>

            <li>
              <FiMessageSquare />
              <span>पूर्वाधार योजना</span>
            </li>
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

        <button className="sidebar-btn logout">
          <FiLogOut />
          <span>Logout</span>
        </button>
      </div>
    </aside>
  );
}