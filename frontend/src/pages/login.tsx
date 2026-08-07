import { useState,useEffect } from "react";
import { Link, useNavigate } from "react-router-dom";
import { FaEye, FaEyeSlash } from "react-icons/fa";
import { handleApiError } from "../utils/hanle_api_errors";
import toast from "react-hot-toast";
import api from "../api/axios";
import "../styles/auth.css";

export default function Login() {
  const navigate = useNavigate();

  const [showPassword, setShowPassword] = useState(false);

  const [formData, setFormData] = useState({
    email: "",
    password: "",
  });


  useEffect(() => {

  const params = new URLSearchParams(window.location.search);

  if (params.get("expired")) {

    toast.error("Session expired. Please login again.");

    window.history.replaceState({}, "", "/login");

  }

  }, []);

  const handleChange = (
    e: React.ChangeEvent<HTMLInputElement>
  ) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value,
    });
  };

  const handleLogin = async (
    e: React.FormEvent
  ) => {
    e.preventDefault();

    //connect with the backend 
    try {
      const response = await api.post(
        "/auth/login",
        formData
      );

      // Save JWT Token
      localStorage.setItem(
        "token",
        response.data.access_token
      );

      // Save Username
      localStorage.setItem(
        "username",
        response.data.user.name
      );


      toast.success(response.data.message);

      setTimeout(() => {
        navigate("/chat");
      }, 1500);

    } catch (error) {
      handleApiError(error);
    }
  };

  return (
    <div className="container">
      <div className="login-box">

        <form onSubmit={handleLogin}>

          <div className="input-group">
            <label>Email</label>

            <input
              type="email"
              name="email"
              placeholder="Enter your email"
              value={formData.email}
              onChange={handleChange}
              required
            />
          </div>

          <div className="input-group">
            <label>Password</label>

            <div className="password-input">

              <input
                type={
                  showPassword
                    ? "text"
                    : "password"
                }
                name="password"
                placeholder="Enter your password"
                value={formData.password}
                onChange={handleChange}
                required
              />

              <button
                type="button"
                className="eye-btn"
                onClick={() =>
                  setShowPassword(!showPassword)
                }
              >
                {showPassword ? (
                  <FaEyeSlash />
                ) : (
                  <FaEye />
                )}
              </button>

            </div>
          </div>

          <button
            type="submit"
            className="login-btn"
          >
            Login
          </button>

        </form>

        <p className="signup-text">
          Don't have an account?{" "}
          <Link to="/signup">
            Sign Up
          </Link>
        </p>

      </div>
    </div>
  );
}