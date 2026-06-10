#api entrypoint (FastAPI)

from app.rag_pipeline import rag_pipeline


def main():
    print("RAG Chatbot Started \n")

    session_id = input("Input the session id/user_name: ")

    while True:
        user_query = input(f"\n {session_id}:  ")

        if user_query.lower in ["exit","quit"]:
            print("Thankyou! GoodDay")

        else:
            response = rag_pipeline(user_query,session_id)

            print("\n Bot:", response)

if __name__ == "__main__":
    main()