from environments.fourroom import FourRoom


class SixRoom(FourRoom):

    def __init__(self):
        super().__init__()

        self.height = 23
        self.width = 30

        self.rooms = [{'corner': (0, 0),
                       'width': 8,
                       'height': 11,
                       'corridors': [(8, 5), (3, 11)]},
                      {'corner': (9, 0),
                       'width': 9,
                       'height': 12,
                       'corridors': [(8, 5), (13, 12), (18, 4)]},
                      {'corner': (19, 0),
                       'width': 11,
                       'height': 10,
                       'corridors': [(18, 4), (24, 10)]},
                      {'corner': (19, 11),
                       'width': 11,
                       'height': 12,
                       'corridors': [(24, 10), (18, 17)]},
                      {'corner': (9, 13),
                       'width': 9,
                       'height': 10,
                       'corridors': [(18, 17), (13, 12), (8, 18)]},
                      {'corner': (0, 12),
                       'width': 8,
                       'height': 11,
                       'corridors': [(8, 18), (8, 5)]}
                      ]

        self.goals = [(room['corner'][0] + room['width'] // 2, room['corner'][1] + room['height'] // 2)
                      for room in self.rooms]

        self.encoded_state_len = self.width + self.height
        return
