import { AxiosError } from "axios";
import toast from "react-hot-toast";

export const handleApiError = (error: unknown) => {

  if (error instanceof AxiosError) {

    // Backend responded => handle error occured from backend
    if (error.response) {

      toast.error(
        error.response.data?.message || "Something went wrong."
      );

      return;
    }

    // Request sent but no response => if there is no response from backend
    if (error.request) {

      toast.error("Unable to connect to the server.");

      return;
    }
  }

  // Unexpected error
  toast.error("Unexpected error occurred.");
};