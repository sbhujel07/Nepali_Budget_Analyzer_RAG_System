import { FiBell, FiUser } from "react-icons/fi";

export default function Navbar() {
  return (
    <header className="navbar">

      <div className="navbar-title">

        <h1>Nepal Annual Budget Assistant</h1>

        <p>AI-powered budget information system</p>
      </div>

      <div className="navbar-actions">

        <button className="icon-btn">
          <FiBell size={20} />
        </button>

        <div className="profile">

          <FiUser size={20} />

          <span>Sandip</span>

        </div>

      </div>

    </header>
  );
}