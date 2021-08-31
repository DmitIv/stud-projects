class FieldDescription:
    @staticmethod
    def get_description() -> dict:
        game_field: dict = {
            # _____________
            # main diagonal
            # _____________
            (1, 1): {
                "weight": 0,
                "limit": 2,
                "reachable": ((1, 4), (4, 1), (), ()),
                "owner": 0
            },
            (2, 2): {
                "weight": 0,
                "limit": 2,
                "reachable": ((2, 4), (4, 2), (), ()),
                "owner": 0
            },
            (3, 3): {
                "weight": 0,
                "limit": 2,
                "reachable": ((3, 4), (4, 3), (), ()),
                "owner": 0
            },
            (5, 5): {
                "weight": 0,
                "limit": 2,
                "reachable": ((), (), (5, 4), (4, 5)),
                "owner": 0
            },
            (6, 6): {
                "weight": 0,
                "limit": 2,
                "reachable": ((), (), (6, 4), (4, 6)),
                "owner": 0
            },
            (7, 7): {
                "weight": 0,
                "limit": 2,
                "reachable": ((), (), (7, 4), (4, 7)),
                "owner": 0
            },
            # ___________________
            # additional diagonal
            # ___________________
            (1, 7): {
                "weight": 0,
                "limit": 2,
                "reachable": ((), (4, 7), (1, 4), ()),
                "owner": 0
            },
            (2, 6): {
                "weight": 0,
                "limit": 2,
                "reachable": ((), (4, 6), (2, 4), ()),
                "owner": 0
            },
            (3, 5): {
                "weight": 0,
                "limit": 2,
                "reachable": ((), (4, 5), (3, 4), ()),
                "owner": 0
            },
            (5, 3): {
                "weight": 0,
                "limit": 2,
                "reachable": ((5, 4), (), (), (4, 3)),
                "owner": 0
            },
            (6, 2): {
                "weight": 0,
                "limit": 2,
                "reachable": ((6, 4), (), (), (4, 2)),
                "owner": 0
            },
            (7, 1): {
                "weight": 0,
                "limit": 2,
                "reachable": ((7, 4), (), (), (4, 1)),
                "owner": 0
            },
            # _____________
            # perpendicular
            # _____________
            (1, 4): {
                "weight": 0,
                "limit": 3,
                "reachable": ((1, 7), (2, 4), (1, 1), ()),
                "owner": 0
            },
            (2, 4): {
                "weight": 0,
                "limit": 4,
                "reachable": ((2, 6), (3, 4), (2, 2), (1, 4)),
                "owner": 0
            },
            (3, 4): {
                "weight": 0,
                "limit": 3,
                "reachable": ((3, 5), (), (3, 3), (2, 4)),
                "owner": 0
            },
            (5, 4): {
                "weight": 0,
                "limit": 3,
                "reachable": ((5, 5), (6, 4), (5, 3), ()),
                "owner": 0
            },
            (6, 4): {
                "weight": 0,
                "limit": 4,
                "reachable": ((6, 6), (7, 4), (6, 2), (5, 4)),
                "owner": 0
            },
            (7, 4): {
                "weight": 0,
                "limit": 3,
                "reachable": ((7, 7), (), (7, 1), (6, 4)),
                "owner": 0
            },
            # __________
            # horizontal
            # __________
            (4, 1): {
                "weight": 0,
                "limit": 3,
                "reachable": ((4, 2), (7, 1), (), (1, 1)),
                "owner": 0
            },
            (4, 2): {
                "weight": 0,
                "limit": 4,
                "reachable": ((4, 3), (6, 2), (4, 1), (2, 2)),
                "owner": 0
            },
            (4, 3): {
                "weight": 0,
                "limit": 3,
                "reachable": ((), (5, 3), (4, 2), (3, 3)),
                "owner": 0
            },
            (4, 5): {
                "weight": 0,
                "limit": 3,
                "reachable": ((4, 6), (5, 5), (), (3, 5)),
                "owner": 0
            },
            (4, 6): {
                "weight": 0,
                "limit": 4,
                "reachable": ((4, 7), (6, 6), (4, 5), (2, 6)),
                "owner": 0
            },
            (4, 7): {
                "weight": 0,
                "limit": 3,
                "reachable": ((), (7, 7), (4, 6), (1, 7)),
                "owner": 0
            }
        }
        special_point: dict = {
            (0, 0): {
                "weight": -100,
                "limit": 100,
                "reachable": tuple(game_field.keys()),
                "owner": 3
            }
        }
        return {**game_field, **special_point}