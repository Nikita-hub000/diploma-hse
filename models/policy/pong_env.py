# pong_env.py
import numpy as np
import random
import sys

class PongEnv:
    def __init__(self, width=600, height=400, paddle_height=60, paddle_width=10, ball_radius=8):
        self.width = width
        self.height = height
        self.paddle_height = paddle_height
        self.paddle_width = paddle_width
        self.ball_radius = ball_radius

        self.norm_width = self.width / 2.0
        self.norm_height = self.height / 2.0

        self.ball_x = 0.0; self.ball_y = 0.0
        self.ball_vx = 0.0; self.ball_vy = 0.0
        self.paddle1_y = 0.0 
        self.paddle2_y = 0.0 

        self.paddle_speed = 12.0
        self.ball_speed_initial = 10.0
        self.ball_speed_increase = 0.4
        self.current_ball_speed = self.ball_speed_initial
        self.max_ball_speed = 24.0
        self.opponent_speed_limit = 12.0

        self.state_dim = 5
        self.action_dim = 3
        self.screen = None
        self.clock = None

        self.reset()

    def _normalize_state(self):
        """ Normalizes the current game state for the agent (5 dimensions). """
        state = [
            self.ball_x / self.norm_width,
            self.ball_y / self.norm_height,
            self.ball_vx / self.max_ball_speed,
            self.ball_vy / self.max_ball_speed,
            self.paddle1_y / self.norm_height,
        ]
        return np.array(state, dtype=np.float32)

    def reset(self):
        self.ball_x = 0.0
        self.ball_y = random.uniform(-self.norm_height * 0.5, self.norm_height * 0.5)
        angle = random.uniform(-np.pi / 4, np.pi / 4)
        direction = 1 if random.random() < 0.5 else -1
        self.current_ball_speed = self.ball_speed_initial
        self.ball_vx = direction * self.current_ball_speed * np.cos(angle)
        self.ball_vy = self.current_ball_speed * np.sin(angle)
        self.paddle1_y = 0.0
        self.paddle2_y = 0.0
        observation = self._normalize_state()
        info = {}
        return observation, info

    def step(self, action):
        if action == 1: self.paddle1_y -= self.paddle_speed
        elif action == 2: self.paddle1_y += self.paddle_speed
        paddle_half_height = self.paddle_height / 2.0
        self.paddle1_y = np.clip(self.paddle1_y, -self.norm_height + paddle_half_height, self.norm_height - paddle_half_height)

        target_y = self.paddle2_y
        if self.ball_y > self.paddle2_y:
             move_amount = min(self.opponent_speed_limit, self.ball_y - self.paddle2_y)
             target_y += move_amount
        elif self.ball_y < self.paddle2_y:
             move_amount = min(self.opponent_speed_limit, self.paddle2_y - self.ball_y)
             target_y -= move_amount
        self.paddle2_y = np.clip(target_y, -self.norm_height + paddle_half_height, self.norm_height - paddle_half_height)

        prev_ball_x = self.ball_x
        prev_ball_y = self.ball_y

        self.ball_x += self.ball_vx
        self.ball_y += self.ball_vy

        reward = 0.0
        terminated = False
        paddle_hit = False 


        if self.ball_y - self.ball_radius < -self.norm_height or self.ball_y + self.ball_radius > self.norm_height:
            self.ball_vy *= -1
            self.ball_y = np.clip(self.ball_y, -self.norm_height + self.ball_radius, self.norm_height - self.ball_radius)

        paddle1_front_edge = -self.norm_width + self.paddle_width
        paddle2_front_edge = self.norm_width - self.paddle_width 

        if self.ball_vx < 0 and \
           prev_ball_x - self.ball_radius >= paddle1_front_edge and \
           self.ball_x - self.ball_radius < paddle1_front_edge and \
           abs(self.ball_y - self.paddle1_y) < paddle_half_height + self.ball_radius: 

            self.ball_vx *= -1
            relative_intersect_y = (self.paddle1_y - self.ball_y) / paddle_half_height
            bounce_angle = relative_intersect_y * (np.pi / 3)
            self.current_ball_speed = min(self.current_ball_speed + self.ball_speed_increase, self.max_ball_speed)
            self.ball_vx = self.current_ball_speed * np.cos(bounce_angle)
            self.ball_vy = self.current_ball_speed * -np.sin(bounce_angle)
            self.ball_x = paddle1_front_edge + self.ball_radius
            paddle_hit = True

     
        elif self.ball_vx > 0 and \
             prev_ball_x + self.ball_radius <= paddle2_front_edge and \
             self.ball_x + self.ball_radius > paddle2_front_edge and \
             abs(self.ball_y - self.paddle2_y) < paddle_half_height + self.ball_radius: 

            self.ball_vx *= -1
            relative_intersect_y = (self.paddle2_y - self.ball_y) / paddle_half_height
            bounce_angle = relative_intersect_y * (np.pi / 3)
            self.current_ball_speed = min(self.current_ball_speed + self.ball_speed_increase, self.max_ball_speed)
            self.ball_vx = -self.current_ball_speed * np.cos(bounce_angle)
            self.ball_vy = self.current_ball_speed * -np.sin(bounce_angle)
            self.ball_x = paddle2_front_edge - self.ball_radius
            paddle_hit = True


        if not paddle_hit:
            if self.ball_x - self.ball_radius < -self.norm_width: 
                reward = -1.0
                terminated = True
            elif self.ball_x + self.ball_radius > self.norm_width: 
                reward = 1.0
                terminated = True

        
        observation = self._normalize_state()
        truncated = False
        info = {}
        return observation, reward, terminated, truncated, info

    def render(self, mode='human'):
        try:
            import pygame
        except ImportError:
            print("Pygame not installed, skipping rendering. Run 'pip install pygame'")
            return

        screen_width = int(self.width)
        screen_height = int(self.height)

        if self.screen is None:
             try:
                 pygame.display.init()
                 if not pygame.display.get_init():
                     print("Warning: No display available for Pygame rendering.")
                     self.screen = None
                     return
                 self.screen = pygame.display.set_mode((screen_width, screen_height))
                 pygame.display.set_caption("Pong RL")
                 self.clock = pygame.time.Clock()
             except pygame.error as e:
                 print(f"Pygame display init error: {e}. Rendering disabled.")
                 self.screen = None 
                 return 

        if self.screen is None:
            return

        try:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close() 
                    raise SystemExit("Pygame window closed by user.")
        except SystemExit: 
             raise 
        except pygame.error as e:
            print(f"Pygame event error: {e}. Stopping rendering.")
            self.close()
            return

        def to_screen_pos(x, y):
            sx = int(x + self.norm_width)
            sy = int(y + self.norm_height)
            return sx, sy

        self.screen.fill((0, 0, 0))
        paddle_half_height = self.paddle_height / 2.0
        paddle1_rect = pygame.Rect(0, 0, self.paddle_width, self.paddle_height); paddle1_rect.center = to_screen_pos(-self.norm_width + self.paddle_width / 2, self.paddle1_y); pygame.draw.rect(self.screen, (255, 255, 255), paddle1_rect)
        paddle2_rect = pygame.Rect(0, 0, self.paddle_width, self.paddle_height); paddle2_rect.center = to_screen_pos(self.norm_width - self.paddle_width / 2, self.paddle2_y); pygame.draw.rect(self.screen, (255, 255, 255), paddle2_rect)
        ball_sx, ball_sy = to_screen_pos(self.ball_x, self.ball_y); pygame.draw.circle(self.screen, (255, 255, 255), (ball_sx, ball_sy), self.ball_radius)
        center_x = screen_width // 2
        for y in range(0, screen_height, 20): pygame.draw.rect(self.screen, (100, 100, 100), (center_x - 2, y, 4, 10))
        pygame.display.flip()
        if self.clock: self.clock.tick(120) 

    def close(self):
        if self.screen is not None:
            try:
                import pygame
                pygame.display.quit()
                pygame.quit()
            except ImportError: pass
            except Exception as e: print(f"Error closing pygame: {e}")
            finally: self.screen = None; self.clock = None
