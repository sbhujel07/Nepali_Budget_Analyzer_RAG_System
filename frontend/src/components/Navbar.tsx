import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { FiBell, FiUser } from "react-icons/fi";

export default function Navbar() {
  const username = localStorage.getItem("username");

  const navigate = useNavigate();

  const [showMenu, setShowMenu] = useState(false);

  const handleLogout = () => {
    localStorage.removeItem("token");
    localStorage.removeItem("username");

    navigate("/login", { replace: true });
  };

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

        <div className="profile-container">

          <div
            className="profile"
            onClick={() => setShowMenu(!showMenu)}
          >
            <FiUser size={20} />

            <span>{username || "User"}</span>
          </div>

          {showMenu && (
            <div className="profile-menu">

              <button onClick={handleLogout}>
                Logout
              </button>

            </div>
          )}

        </div>

      </div>

    </header>
  );
}