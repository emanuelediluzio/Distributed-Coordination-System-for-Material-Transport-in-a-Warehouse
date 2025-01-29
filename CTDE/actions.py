from enum import Enum

class Action(Enum):
    NOOP = 0
    FORWARD = 1
    LEFT = 2
    RIGHT = 3
    TOGGLE_LOAD = 4
    SHIFT_BATTERY = 5
    SET_MESSAGE_BIT0 = 6
    SET_MESSAGE_BIT1 = 7
    CLEAR_MESSAGE = 8
    # Azioni aggiuntive per comunicazione avanzata
    SEND_BATTERY_STATUS = 9
    SEND_PROXIMITY_INFO = 10
    SEND_CHARGING_STATION_INFO = 11
    REQUEST_HELP = 12
    SEND_TASK_ASSIGNMENT = 13
    SEND_PATH_BLOCKED = 14
    SEND_PRIORITY_UPDATE = 15