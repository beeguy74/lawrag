from os import getenv
from dotenv import load_dotenv

if __name__ == '__main__':
    load_dotenv()
    while True:
        question = input("\033[92mPosez votre question par rapport au code du travail ou tapez 'q' pour quitter: \033[0m")
        if question == 'q':
            break
        # on cherche la reponse dans le code du travail
        print(f"La reponse a la question '{question}' est: ...")
