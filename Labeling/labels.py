from labeling import labeling, VideoPlayer

def main() -> None:
    l = labeling('Walking')
    # print(l.get_insert_labels())
    # print(l.classify())
    # print(l.labels)
    
    vid = VideoPlayer()
    vid.play_vid(r'SomeVid.mp4', 0, 2)
    l.classify()
    # vid.play_vid(r'Timer2.mp4', 299, 10)




if __name__ == '__main__':
    main()
    