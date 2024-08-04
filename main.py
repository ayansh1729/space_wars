import pygame
import random
import numpy as np
import pygame.mixer
# import tensorflow as tf

WIDTH, HEIGHT = 900, 600
VEL = 5
BULLET_VEL = 10
MAX_BULLETS = 5
NUM_STATES = 20
NUM_ACTIONS = 5
PENALTY_FACTOR = 0.9
TIME_THRESHOLD = 100

WHITE = (255, 255, 255)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
BLACK = (0, 0, 0)

YELLOW_HIT = pygame.USEREVENT + 1
RED_HIT = pygame.USEREVENT + 2

pygame.init()
pygame.mixer.init()

bullet_fire_sound = pygame.mixer.Sound("./Assets/bullet_fire_sound.mp3")
bullet_hit_sound = pygame.mixer.Sound("./Assets/bullet_hit_sound.mp3")


jet_left_img = pygame.image.load("./Assets/jet_left.png")
jet_right_img = pygame.image.load("./Assets/jet_right.png")
bg_img = pygame.image.load("./Assets/space.png")
bg_img = pygame.transform.scale(bg_img, (WIDTH, HEIGHT))

JET_WIDTH, JET_HEIGHT = 55, 44
jet_left_img = pygame.transform.scale(jet_left_img, (JET_WIDTH, JET_HEIGHT))
jet_right_img = pygame.transform.scale(jet_right_img, (JET_WIDTH, JET_HEIGHT))

# Flip the images
jet_left_img = pygame.transform.flip(jet_left_img, True, False)
jet_right_img = pygame.transform.flip(jet_right_img, True, False)

# Rotate the images
jet_left_img = pygame.transform.rotate(jet_left_img, 90)  # Rotate left image counterclockwise by 90 degrees
jet_right_img = pygame.transform.rotate(jet_right_img, -90) 

bullet_img = pygame.image.load("./Assets/bullet.png")
bullet_img = pygame.transform.scale(bullet_img, (10, 5))

WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Fighter Jet PvP")

class Environment:
    def __init__(self):
        self.yellow_health = 5
        self.red_health = 5

    def update(self, yellow, red, yellow_bullets, red_bullets, yellow_action, red_action):
        self.update_yellow(yellow, yellow_action)
        self.update_red(red, red_action)

        if yellow_action == 4 and len(yellow_bullets) < MAX_BULLETS:
            bullet = [yellow.x + yellow.width, yellow.y + yellow.height // 2 - 2, False]  # Bullet fired by yellow jet
            yellow_bullets.append(bullet)

        if red_action == 4 and len(red_bullets) < MAX_BULLETS:
            bullet = [red.x + red.width, red.y + red.height // 2 - 2, True]  # Bullet fired by red jet
            red_bullets.append(bullet)

        for bullet in yellow_bullets:
            bullet[0] += BULLET_VEL
            if red.colliderect(pygame.Rect(bullet[0], bullet[1], bullet_img.get_width(), bullet_img.get_height())):
                pygame.event.post(pygame.event.Event(RED_HIT))
                bullet_hit_sound.play()
                print("red hit")
                self.red_health -= 1
                yellow_bullets.remove(bullet)
            elif bullet[0] > WIDTH:
                yellow_bullets.remove(bullet)

        for bullet in red_bullets:
            bullet[0] -= BULLET_VEL
            if yellow.colliderect(pygame.Rect(bullet[0], bullet[1], bullet_img.get_width(), bullet_img.get_height())):
                pygame.event.post(pygame.event.Event(YELLOW_HIT))
                bullet_hit_sound.play()
                print("yellow hit")
                self.yellow_health -= 1
                red_bullets.remove(bullet)
            elif bullet[0] < 0:
                red_bullets.remove(bullet)

    def update_yellow(self, jet, action):
        if action == 0:
            if jet.y - VEL > 0:
                jet.y -= VEL
        elif action == 1:
            if jet.y + jet.height + VEL < HEIGHT - 10:
                jet.y += VEL

        elif action == 2:
            if jet.x - VEL > 0:
                jet.x -= VEL
        elif action == 3:
            if jet.x + VEL + jet.width < WIDTH // 2 - 10:
                jet.x += VEL
    
    def update_red(self, jet, action):
        if action == 2:
            if jet.y - VEL > 0:
                jet.y -= VEL
        elif action == 1:
            if jet.y + jet.height + VEL < HEIGHT - 10:
                jet.y += VEL
        elif action == 0:
            if jet.x - VEL > WIDTH//2 + 10:
                jet.x -= VEL
        elif action == 3:
            if jet.x + VEL + jet.width < WIDTH // 2 - 10:
                jet.x += VEL

    def get_state(self, jet):
        state_x = min(int(jet.x / (WIDTH / NUM_STATES)), NUM_STATES - 1)
        state_y = min(int(jet.y / (HEIGHT / NUM_STATES)), NUM_STATES - 1)
        return state_x, state_y

    def get_reward(self, yellow, red):
        reward = 0
        if pygame.event.get(RED_HIT):
            reward = -100
        elif pygame.event.get(YELLOW_HIT):
            reward = 1000
        else:
            reward = -int((yellow.y-red.y)**2 + (yellow.x-red.x)**2)
        return reward

    def reset(self):
        self.yellow_health = 5
        self.red_health = 5

class Agent:
    def __init__(self):
    # Assuming NUM_STATES and NUM_ACTIONS are defined elsewhere
        self.q_table = np.zeros((NUM_STATES, NUM_STATES, NUM_STATES, NUM_STATES, NUM_STATES, NUM_STATES, NUM_ACTIONS))
        self.epsilon = 0.5  # Start with a high epsilon for maximum explorationss
        self.min_epsilon = 0.3  # Set a low minimum epsilon to continue exploration
        self.decay_rate = 0.995  # Slow down the decay rate to extend exploration
        self.learning_rate = 0.2
        self.discount_factor = 0.9
        self.state_time_counters = np.zeros((NUM_STATES, NUM_STATES, NUM_STATES, NUM_STATES))  # Time counters for each state

    def select_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            action = random.choice(range(NUM_ACTIONS))
        else:
            action = np.argmax(self.q_table[state])
        return action

    def update(self, state, other_jet_position, action, reward, next_state):
        old_q_value = self.q_table[state][other_jet_position][action]
        max_next_q_value = np.max(self.q_table[next_state])
        new_q_value = old_q_value + self.learning_rate * (reward + self.discount_factor * max_next_q_value - old_q_value)
        self.q_table[state][other_jet_position][action] = new_q_value

    def learn(self):
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.decay_rate

    def penalty(self, state, other_jet_position):
        # Penalize the agent for spending too much time in the state
        self.q_table[state][other_jet_position] *= PENALTY_FACTOR
        # Reset the time counter for the state
        self.state_time_counters[state][other_jet_position] = 0


#Implementing DLQ here
# class Agent:
#     def __init__(self, state_size, action_size):
#         self.state_size = state_size
#         self.action_size = action_size
#         self.epsilon = 1.0
#         self.min_epsilon = 0.01
#         self.epsilon_decay = 0.995
#         self.gamma = 0.99  # discount factor
#         self.learning_rate = 0.001
#         self.batch_size = 64
#         self.memory = []

#         # Build Q-network
#         self.model = self._build_model()

#     def _build_model(self):
#         model = tf.keras.Sequential([
#             tf.keras.layers.Dense(64, activation='relu', input_shape=(self.state_size,)),
#             tf.keras.layers.Dense(64, activation='relu'),
#             tf.keras.layers.Dense(self.action_size)  # Output layer, no activation (linear output)
#         ])
#         model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate))
#         return model

#     def remember(self, state, action, reward, next_state, done):
#         self.memory.append((state, action, reward, next_state, done))

#     def act(self, state):
#         if np.random.rand() <= self.epsilon:
#             return random.randrange(self.action_size)
#         else:
#             q_values = self.model.predict(state)
#             return np.argmax(q_values[0])

#     def replay(self):
#         if len(self.memory) < self.batch_size:
#             return
#         minibatch = random.sample(self.memory, self.batch_size)
#         for state, action, reward, next_state, done in minibatch:
#             target = reward
#             if not done:
#                 target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
#             target_f = self.model.predict(state)
#             target_f[0][action] = target
#             self.model.fit(state, target_f, epochs=1, verbose=0)
#         if self.epsilon > self.min_epsilon:
#             self.epsilon *= self.epsilon_decay

def render(yellow, red, yellow_bullets, red_bullets, yellow_health, red_health):
    WIN.blit(bg_img, (0, 0))
    pygame.draw.rect(WIN, BLACK, (WIDTH // 2 - 5, 0, 10, HEIGHT))
    WIN.blit(jet_left_img, (yellow.x, yellow.y))
    WIN.blit(jet_right_img, (red.x, red.y))
    for bullet in yellow_bullets:
        bullet_img_rotated = pygame.transform.scale(bullet_img, (10, 10))  # Scale up bullet image
        bullet_rect = bullet_img_rotated.get_rect(center=(bullet[0], bullet[1]))
        bullet_fire_sound.play()
        WIN.blit(bullet_img_rotated, bullet_rect)
    for bullet in red_bullets:
        bullet_img_rotated = pygame.transform.flip(pygame.transform.scale(bullet_img, (10, 10)), True, False)  # Scale up and flip bullet image
        bullet_rect = bullet_img_rotated.get_rect(center=(bullet[0], bullet[1]))
        bullet_fire_sound.play()
        WIN.blit(bullet_img_rotated, bullet_rect)
    yellow_health_text = pygame.font.SysFont("comicsans", 40).render(f"Health: {yellow_health}", True, BLACK)
    red_health_text = pygame.font.SysFont("comicsans", 40).render(f"Health: {red_health}", True, BLACK)
    WIN.blit(yellow_health_text, (10, 10))
    WIN.blit(red_health_text, (WIDTH - red_health_text.get_width() - 10, 10))
    pygame.display.update()

    
def main():
    clock = pygame.time.Clock()
    environment = Environment()
    agent = Agent()

    yellow = pygame.Rect(100, 100, JET_WIDTH, JET_HEIGHT)
    red = pygame.Rect(700, 100, JET_WIDTH, JET_HEIGHT)
    yellow_bullets = []
    red_bullets = []

    run = True
    while run:
        clock.tick(60)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        keys = pygame.key.get_pressed()
        yellow_action = -1
        if keys[pygame.K_w]:
            yellow_action = 0
        elif keys[pygame.K_s]:
            yellow_action = 1
        elif keys[pygame.K_a]:
            yellow_action = 2
        elif keys[pygame.K_d]:
            yellow_action = 3
        elif keys[pygame.K_LSHIFT]:
            yellow_action = 4

        if event.type == RED_HIT:
            environment.red_health -= 1

        if event.type == YELLOW_HIT:
            environment.yellow_health -= 1

        yellow_state = environment.get_state(yellow)
        print("Yellow: ",yellow_state)
        red_state = environment.get_state(red)
        print(red_state)
        agent.state_time_counters[yellow_state] += 1
        agent.state_time_counters[red_state] += 1

        # Check if time threshold is exceeded for yellow jet's state
        if agent.state_time_counters[yellow_state].any() > TIME_THRESHOLD:
            agent.penalty(yellow_state)

        # Check if time threshold is exceeded for red jet's state
        if agent.state_time_counters[red_state].any() > TIME_THRESHOLD:
            agent.penalty(red_state)

        # Select action for the red jet based on its state
        red_action = agent.select_action(red_state)

        # Update the environment with actions of both jets
        environment.update(yellow, red, yellow_bullets, red_bullets, yellow_action, red_action)

        # Get reward from the environment
        reward = environment.get_reward(yellow,red)

        # Get next state of the yellow jet
        next_yellow_state = environment.get_state(yellow)

        # Update the Q-table with the yellow jet's action, reward, and next state
        agent.update(yellow_state, environment.get_state(yellow), yellow_action, reward, next_yellow_state)

        # Let the agent learn from its experience
        agent.learn()

        # Check if any jet's health is zero
        if environment.red_health <= 0 or environment.yellow_health <= 0:
            run = False

        # Render the game with updated positions and health
        render(yellow, red, yellow_bullets, red_bullets, environment.yellow_health, environment.red_health)

    pygame.quit()


if __name__ == "__main__":
    main()
