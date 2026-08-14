import axios from "axios";

console.log("API URL:", import.meta.env.VITE_API_URL);

//create axios object for api call
const api = axios.create({
    baseURL: import.meta.env.VITE_API_URL,
    timeout: 10000,
    headers: {
        "Content-Type": "application/json",
    },
}
);




//Request Interceptor => it is automatically used by axios using request.use during api call
api.interceptors.request.use(
  (config) => {

    const token = localStorage.getItem("token");

    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }

    return config;
  },

  (error) => {
    return Promise.reject(error);
  }
);




// Response Interceptor -> used to show the error in frontend
api.interceptors.response.use(

  (response) => {
    return response;
  },

  (error) => {

    if (error.response?.status === 401   && !error.config?.url?.includes("/auth/login")  ) {

      localStorage.removeItem("token");

      window.location.href = "/login?expired=true";
    }

    return Promise.reject(error);
  }

);

export default api;