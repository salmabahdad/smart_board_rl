import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from enum import Enum
import numpy as np

register(
    id='smart-board',
    entry_point='smart_board_env:AirplaneEnv', # module:class
)

class PassengerStatus(Enum):
    WALKING_TO_SEAT  = 0   
    WAITING_IN_AISLE  = 1  
    STOWING_SUITCASE  = 2   
    SEATED_AND_READY  = 3  
    # Returns the string representation of the PassengerStatus enum.
    def __str__(self):
     mapping = {
        PassengerStatus.WALKING_TO_SEAT: "WALKING_TO_SEAT",
        PassengerStatus.WAITING_IN_AISLE: "WAITING_IN_AISLE",
        PassengerStatus.STOWING_SUITCASE: "STOWING_SUITCASE",
        PassengerStatus.SEATED_AND_READY: "SEATED_AND_READY"
    }
     return mapping.get(self, "UNKNOWN_STATUS")


class AirplaneEnv(gym.Env):
    metadata = {'render_modes': ['human','terminal'], 'render_fps': 1}

    def __init__(self, render_mode=None, seats_row=4 ,rows_num=10):
       self.seats_row=seats_row 
       self.rows_num=rows_num
       self.total_seats=seats_row*rows_num
       self.render_mode=render_mode
       self.action_space = spaces.Discrete(self.rows_num)
       # [0,1,1,2,2,3...]
       self.observation_space = spaces.Box(
            low=-1,
            high=self.total_seats-1,
            shape=(self.total_seats * 2,),
            dtype=np.int32
        )
 

    def reset(self, seed=None, options=None):
        super().reset(seed=seed) 

        self.airplane_rows = [AirplaneRow(row_num, self.seats_row) for row_num in range(self.rows_num)]
        self.lobby = Lobby(self.rows_num, self.seats_row)
        self.boarding_line = BoardingLine(self.rows_num)

        self.render()

        return self._get_observation(), {}
    
    def _get_observation(self):
        observation = []
        for passenger in self.boarding_line.line:

            if passenger is None:
                observation.append(-1)
                observation.append(-1)
            else:
                observation.append(passenger.seat_id)
                observation.append(passenger.status.value)

        for i in range(len(self.boarding_line.line), self.total_seats):
            observation.append(-1)
            observation.append(-1)

        return np.array(observation, dtype=np.int32)
    
    def step(self, row_id):
        assert row_id>=0 and row_id<self.rows_num, f"Invalid row number {row_id}"

        reward = 0

        passenger = self.lobby.remove_passenger(row_id)
        self.boarding_line.add_passenger(passenger)

        # If there are passengers in the lobby
        if self.lobby.count_passengers()>0:
            self._move()
            reward = self._calculate_reward()
        else:
            # No more passengers in the lobby
            while self.is_onboarding():
                self._move()
                reward += self._calculate_reward()

        if self.is_onboarding():
            terminated = False
        else:
            terminated = True

        return self._get_observation(), reward, terminated, False, {}

    def _calculate_reward(self):
        reward = -self.boarding_line.num_passengers_WAITING_IN_AISLE() + self.boarding_line.num_passengers_WALKING_TO_SEAT()
        return reward

    def is_onboarding(self):
        # If there are passengers in the lobby or in the boarding line
        if self.lobby.count_passengers() > 0 or self.boarding_line.is_onboarding():
            return True

        return False

    def _move(self):

        for row_num, passenger in enumerate(self.boarding_line.line):
            if passenger is None:
                continue

            # If outside of airplane's aisle
            if row_num >= len(self.airplane_rows):
                break

            # Try to sit passenger if done remove from line
            if self.airplane_rows[row_num].try_sit_passenger(passenger):
                self.boarding_line.line[row_num] = None

        # Move forward
        self.boarding_line.move_forward()

        self.render()


    def render(self):
        if self.render_mode is None:
            return

        if self.render_mode == 'terminal':
            self._render_terminal()


    def _render_terminal(self):
        print("Seats".center(19) + " | Aisle Line")
        for row in self.airplane_rows:
            for seat in row.seats:
                print(seat, end=" ")

            if row.row_num < len(self.boarding_line.line):
                passenger = self.boarding_line.line[row.row_num]

                status = "" if passenger is None else passenger.status

                print(f"| {passenger} {status}", end=" ")

            print()

        print("\nLine entering plane:")
        for i in range(self.rows_num, len(self.boarding_line.line)):
            passenger = self.boarding_line.line[i]
            print(f"{passenger} {passenger.status}")

        print("\nLobby:")
        for row in self.lobby.lobby_rows:
            for passenger in row.passengers:
                print(passenger, end=" ")

            if(len(row.passengers) > 0):
                print()

        print("\n")

    # This method is to mask the actions used with maskable PPO agents 
    def action_masks(self) -> list[bool]:
        mask = []

        for row in self.lobby.lobby_rows:
            if len(row.passengers) == 0:
                mask.append(False)
            else:
                mask.append(True)

        return mask


class Passenger:
    def __init__(self, id_seat, id_row):
        self.seat_id = id_seat
        self.row_id = id_row
        self.is_holding_suitcase = True
        self.status = PassengerStatus.WALKING_TO_SEAT   
        
   
    def __str__(self):
        return f"P{self.seat_id:02d}"

class LobbyRow:
    def __init__(self, row_id, seats_row):
        self.row_num = row_id
        self.passengers = [Passenger(row_id * seats_row + i, row_id) for i in range(seats_row)]

class Lobby:
    def __init__(self, num_of_rows, seats_per_row):
        self.num_of_rows = num_of_rows
        self.seats_per_row = seats_per_row
        self.lobby_rows = [LobbyRow(row_num, self.seats_per_row) for row_num in range(self.num_of_rows)]

    def remove_passenger(self, row_num):
        passenger = self.lobby_rows[row_num].passengers.pop()
        return passenger

    def count_passengers(self):
        count = 0
        for row in self.lobby_rows:
            count += len(row.passengers)

        return count

class BoardingLine:
    def __init__(self, rows_num):
        self.num_of_rows = rows_num
        self.line = [None for i in range(rows_num)]

    def add_passenger(self, passenger):
        self.line.append(passenger)

    def is_onboarding(self):
        if (len(self.line) > 0 and not all(passenger is None for passenger in self.line)):
            return True

        return False

    def num_passengers_WAITING_IN_AISLE(self):
        count = 0
        for passenger in self.line:
            if passenger is not None and passenger.status == PassengerStatus.WAITING_IN_AISLE:
                count += 1

        return count

    def num_passengers_WALKING_TO_SEAT(self):
        count = 0
        for passenger in self.line:
            if passenger is not None and passenger.status == PassengerStatus.WALKING_TO_SEAT:
                count += 1

        return count

    def move_forward(self):

        for i, passenger in enumerate(self.line):
            # Skip if no passenger in that spot or  passenger is at the front of the line or passenger is stowing suitcase
            if passenger is None or i==0 or passenger.status == PassengerStatus.STOWING_SUITCASE:
                continue

            # Move the passenger if no one is blocking
            if (passenger.status == PassengerStatus. WAITING_IN_AISLE or passenger.status == PassengerStatus.WALKING_TO_SEAT) and self.line[i-1] is None:
                passenger.status = PassengerStatus.WALKING_TO_SEAT
                self.line[i-1] = passenger
                self.line[i] = None
            else:
                passenger.status = PassengerStatus. WAITING_IN_AISLE

        # move the empty spots to the end of the boarding line
        for i in range(len(self.line)-1, self.num_of_rows-1, -1):
            if self.line[i] is None:
                self.line.pop(i)

class Seat:
    def __init__(self, seat_id, row_id):
        self.seat_id = seat_id
        self.row_id = row_id
        self.passenger = None

    def seat_passenger(self, passenger: Passenger):

        assert self.seat_id == passenger.seat_id, "Seat number doesn't match the Passenger"

        if passenger.is_holding_suitcase:
            passenger.status = PassengerStatus.STOWING_SUITCASE
            passenger.is_holding_suitcase = False
            return False
        else:
            self.passenger = passenger
            self.passenger.status = PassengerStatus.SEATED_AND_READY
            return True

    def __str__(self):
        if self.passenger is None:
            return f"S{self.seat_id:02d}"
        else:
            return f"P{self.seat_id:02d}"

class AirplaneRow:
    def __init__(self, row_num, seats_per_row):
        self.row_num = row_num
        self.seats = [Seat(row_num * seats_per_row + i, row_num) for i in range(seats_per_row)]

    def try_sit_passenger(self, passenger: Passenger):
        # Check if passenger's seat is in this row
        found_seats = list(filter(lambda seats: seats.seat_id == passenger.seat_id, self.seats))

        if found_seats:
            found_seat: Seat = found_seats[0]
            return found_seat.seat_passenger(passenger)

        return False

if __name__ == "__main__":
    # my_check_env()

    env = gym.make('smart-board',seats_row=5, rows_num=10,  render_mode='terminal')

    observation, _ = env.reset()
    terminated = False
    total_reward = 0
    step_count = 0

    while not terminated:
        # Choose random action
        action = env.action_space.sample()

        # Skip action if invalid
        masks = env.unwrapped.action_masks()
        if(masks[action]==False):
           continue

        # Perform action
        observation, reward, terminated, _, _ = env.step(action)
        total_reward += reward

        step_count+=1

        print(f"Step {step_count} Action: {action}")
        print(f"Observation: {observation}")
        print(f"Reward: {reward}\n")

    print(f"Total Reward: {total_reward}")