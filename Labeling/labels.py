from labeling import Labeling, VideoPlayer
import time
import os

def main() -> None:
    # l = Labeling('Walking')
    # print(l.get_insert_labels())
    # print(l.classify())
    # print(l.labels)
    
    vid = VideoPlayer()
    # vid.play_vid(r'SomeVid.mp4', 0, 10)
    # l.classify()
    # vid.play_vid(r'Timer2.mp4', 1, 2)
    start = time.time()
    output = vid.make_clip(r'Timer2.mp4', 1, 2)
    print(time.time()-start)
    # time.sleep(5)
    # os.remove(output)


if __name__ == '__main__':
    main()
    