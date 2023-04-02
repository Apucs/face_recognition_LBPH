from face_register import Register
from faces_train import Train_Faces
from faces_recognition import Recognition
import argparse


class Run:
    def __init__(self):
        super(Run, self).__init__()

    @staticmethod
    def start(action):
        print(action)
        if action == "recog":
            fr = Recognition()
            fr.recognition()

        elif action == "train":
            f_train = Train_Faces()
            f_train.train()

        elif action == "reg":
            f_reg = Register()
            f_reg.register()

        else:
            print(f"Provide the valid action term!")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-a", "--action", type=str, default="recog",
                    help="Please choose the action you want to perform")

    args = vars(ap.parse_args())
    print(args)
    Run().start(action = args["action"])


if __name__ == "__main__":
    main()
