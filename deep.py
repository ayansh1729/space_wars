import pygame
import random
import numpy as np
import pygame.mixer
import tensorflow as tf

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

bullet_fire_sound = pygame.mixer.Sound("./Assets/Bullet_fire_sound.mp3")
bullet_hit_sound = pygame.mixer.Sound("./Assets/bullet_hit_sound.mp3")


jet_left_img = pygame.image.load("./Assets/jet_left.png")
jet_right_img = pygame.image.load("./Assets/jet_right.png")
bg_img = pygame.image.load("./Assets/space.jpg")
bg_img = pygame.transform.scale(bg_img, (WIDTH, HEIGHT))

JET_WIDTH, JET_HEIGHT = 55, 44
jet_left_img = pygame.transform.scale(jet_left_img, (JET_WIDTH, JET_HEIGHT))
jet_right_img = pygame.transform.scale(jet_right_img, (JET_WIDTH, JET_HEIGHT))

jet_left_img = pygame.transform.flip(jet_left_img, True, False)
jet_right_img = pygame.transform.flip(jet_right_img, True, False)

bullet_img = pygame.image.load("Textures/bullet.png")
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

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = 1.0
        self.min_epsilon = 0.01
        self.epsilon_decay = 0.995
        self.gamma = 0.99  # discount factor
        self.learning_rate = 0.001
        self.batch_size = 64
        self.memory = []

        # Build Q-network
        self.model = self._build_model()

    def _build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(self.state_size,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(self.action_size)  # Output layer, no activation (linear output)
        ])
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)  # Updated optimizer initialization
        model.compile(loss='mse', optimizer=optimizer)  # Pass optimizer object instead of parameters
        return model


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        else:
            q_values = self.model.predict(state)
            return np.argmax(q_values[0])


    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        states = np.array([transition[0] for transition in minibatch])
        actions = np.array([transition[1] for transition in minibatch])
        rewards = np.array([transition[2] for transition in minibatch])
        next_states = np.array([transition[3] for transition in minibatch])
        dones = np.array([transition[4] for transition in minibatch])

        # Ensure next_states have the correct shape
        expected_shape = (self.batch_size, self.state_size)
        next_states = np.array([state.ravel() if state.shape == expected_shape else np.zeros(self.state_size) for state in next_states])

        # Ensure states have the correct shape
        states = np.array([state.ravel() if state.shape == expected_shape else np.zeros(self.state_size) for state in states])

        targets = rewards + (1 - dones) * self.gamma * np.amax(self.model.predict(next_states), axis=1)

        target_f = self.model.predict(states)
        target_f[np.arange(self.batch_size), actions] = targets
        self.model.fit(states, target_f, epochs=1, verbose=0)

        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay


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
    agent = DQNAgent(state_size=NUM_STATES, action_size=NUM_ACTIONS)

    yellow = pygame.Rect(100, 100, JET_WIDTH, JET_HEIGHT)  # Controlled by you
    red = pygame.Rect(700, 100, JET_WIDTH, JET_HEIGHT)  # Controlled by AI
    yellow_bullets = []
    red_bullets = []

    run = True
    while run:
        clock.tick(60)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        # Handle player input for the yellow jet (controlled by you)
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w] and yellow.y - VEL > 0:
            yellow.y -= VEL
        if keys[pygame.K_s] and yellow.y + yellow.height + VEL < HEIGHT - 10:
            yellow.y += VEL
        if keys[pygame.K_a] and yellow.x - VEL > 0:
            yellow.x -= VEL
        if keys[pygame.K_d] and yellow.x + VEL + yellow.width < WIDTH // 2 - 10:
            yellow.x += VEL
        if keys[pygame.K_LSHIFT] and len(yellow_bullets) < MAX_BULLETS:
            bullet = [yellow.x + yellow.width, yellow.y + yellow.height // 2 - 2, False]  # Bullet fired by yellow jet
            yellow_bullets.append(bullet)

        # Get the current state of the environment for the red jet (controlled by AI)
        red_state = environment.get_state(red)

        # AI agent selects an action based on the current state for the red jet
        red_action = agent.act(np.array(red_state).reshape(1, -1))

        # Update the environment based on the selected actions for both jets
        environment.update(yellow, red, yellow_bullets, red_bullets, None, red_action)  # Pass None for yellow_action

        # Get the reward from the environment
        reward = environment.get_reward(yellow, red)

        # Get the next state of the environment for the red jet
        next_red_state = environment.get_state(red)

        # Store the experience in the agent's memory for the red jet
        agent.remember(red_state, red_action, reward, next_red_state, False)

        # Perform experience replay for the red jet
        agent.replay()

        if environment.red_health <= 0 or environment.yellow_health <= 0:
            run = False

        render(yellow, red, yellow_bullets, red_bullets, environment.yellow_health, environment.red_health)

    pygame.quit()

if __name__ == "__main__":
    main()
