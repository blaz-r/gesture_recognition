from GestureRecognition import GestureRecognition
from HandRenderer import HandRenderer

gestures = GestureRecognition()
renderer = HandRenderer(gestures)

def main():
    while True:
        frame, hand, gesture = gestures.next_frame()

        hands = []
        if hand:
            hands.append(hand)

        frame = renderer.draw(frame, hands, None)
        key = renderer.waitKey(delay=1)
        if key == 27 or key == ord('q'):
            break
    renderer.exit()
    gestures.exit()



if __name__ == "__main__":
    main()