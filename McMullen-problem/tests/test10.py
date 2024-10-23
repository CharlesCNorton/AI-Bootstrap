import pygame
import sys
import random
import json
import os

# Initialize Pygame
pygame.init()
pygame.mixer.init()

# Screen dimensions
SCREEN_WIDTH = 1024
SCREEN_HEIGHT = 768

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
DARK_BLUE = (10, 10, 50)
LIGHT_BLUE = (173, 216, 230)
GOLD = (255, 215, 0)
GRAY = (100, 100, 100)
TRANSPARENT_BLACK = (0, 0, 0, 180)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BROWN = (139, 69, 19)
YELLOW = (255, 255, 0)

# Fonts
pygame.font.init()
FONT = pygame.font.SysFont('arial', 24)
DIALOGUE_FONT = pygame.font.SysFont('arial', 20)
TITLE_FONT = pygame.font.SysFont('arial', 36)

# Create the screen
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Pirate Adventure")

# Clock for controlling frame rate
clock = pygame.time.Clock()
FPS = 60

# Paths
SAVE_FILE = "savegame.json"

# Sound Setup (Placeholder: Using Pygame's beep for sound effects)
# In a full game, you would load actual sound files
def play_sound(frequency=440, duration=100):
    # Generate a simple beep sound
    sample_rate = 44100
    n_samples = int(round(duration * sample_rate / 1000))
    buf = pygame.sndarray.make_sound(
        (4096 * pygame.np.sin(2.0 * pygame.np.pi * frequency * pygame.np.arange(n_samples) / sample_rate)).astype(pygame.np.int16)
    )
    buf.play()

# Player properties
PLAYER_SIZE = 50
PLAYER_SPEED = 5

# Inventory properties
INVENTORY_WIDTH = 300
INVENTORY_HEIGHT = 400

# Dialogue box properties
DIALOGUE_WIDTH = 800
DIALOGUE_HEIGHT = 150

# Puzzle properties
PUZZLE_HINT = "I be thinkin' ye need to find the golden key!"
PUZZLE_ANSWER = "golden key"

# Load or Initialize Save Data
def load_game():
    if os.path.exists(SAVE_FILE):
        with open(SAVE_FILE, 'r') as f:
            return json.load(f)
    else:
        return {
            "current_scene": "ship",
            "inventory": [],
            "puzzles_solved": {}
        }

def save_game(data):
    with open(SAVE_FILE, 'w') as f:
        json.dump(data, f)

# Classes

class Inventory:
    def __init__(self, items=None):
        self.items = items if items else []
        self.visible = False
        self.font = pygame.font.SysFont('arial', 20)

    def toggle(self):
        self.visible = not self.visible
        play_sound(600, 50)  # Placeholder sound

    def add_item(self, item):
        if item not in self.items:
            self.items.append(item)
            play_sound(800, 50)  # Placeholder sound

    def remove_item(self, item):
        if item in self.items:
            self.items.remove(item)
            play_sound(400, 50)  # Placeholder sound

    def has_item(self, item):
        return item in self.items

    def render(self, surface):
        # Draw inventory background
        inventory_surface = pygame.Surface((INVENTORY_WIDTH, INVENTORY_HEIGHT))
        inventory_surface.set_alpha(220)
        inventory_surface.fill(GRAY)
        surface.blit(inventory_surface, (SCREEN_WIDTH - INVENTORY_WIDTH - 20, 20))

        # Render inventory title
        title = FONT.render("Inventory", True, WHITE)
        surface.blit(title, (SCREEN_WIDTH - INVENTORY_WIDTH - 10, 30))

        # Render inventory items
        for idx, item in enumerate(self.items):
            item_text = FONT.render(f"{idx + 1}. {item}", True, WHITE)
            surface.blit(item_text, (SCREEN_WIDTH - INVENTORY_WIDTH - 10, 70 + idx * 30))

class Dialogue:
    def __init__(self, text, options=None, callback=None):
        self.text = text
        self.visible = True
        self.options = options if options else []
        self.callback = callback

    def render(self, surface):
        # Draw dialogue box
        dialogue_surface = pygame.Surface((DIALOGUE_WIDTH, DIALOGUE_HEIGHT))
        dialogue_surface.set_alpha(220)
        dialogue_surface.fill(BLACK)
        surface.blit(dialogue_surface, ((SCREEN_WIDTH - DIALOGUE_WIDTH) // 2, SCREEN_HEIGHT - DIALOGUE_HEIGHT - 20))

        # Render dialogue text
        lines = self.text.split('\n')
        for idx, line in enumerate(lines):
            text_surface = DIALOGUE_FONT.render(line, True, WHITE)
            surface.blit(text_surface, ((SCREEN_WIDTH - DIALOGUE_WIDTH) // 2 + 20, SCREEN_HEIGHT - DIALOGUE_HEIGHT - 10 + idx * 25))

        # Render options if any
        if self.options:
            for idx, option in enumerate(self.options):
                option_text = DIALOGUE_FONT.render(f"{idx + 1}. {option['text']}", True, YELLOW)
                surface.blit(option_text, ((SCREEN_WIDTH - DIALOGUE_WIDTH) // 2 + 20, SCREEN_HEIGHT - DIALOGUE_HEIGHT - 10 + (len(lines) + idx) * 25))

class Puzzle:
    def __init__(self, name, hint, answer, on_solve=None):
        self.name = name
        self.is_solved = False
        self.hint = hint
        self.answer = answer
        self.on_solve = on_solve

    def attempt_solve(self, player_input):
        if player_input.lower() == self.answer.lower():
            self.is_solved = True
            if self.on_solve:
                self.on_solve()
            return True, "Ye unlocked the treasure! Well done, matey!"
        else:
            return False, "Arrr, that's not the right key. Try again!"

class NPC:
    def __init__(self, name, position, dialogues):
        self.name = name
        self.position = position  # (x, y)
        self.dialogues = dialogues
        self.current_dialogue = 0

    def interact(self):
        if self.current_dialogue < len(self.dialogues):
            dialogue_text = self.dialogues[self.current_dialogue]
            self.current_dialogue += 1
            return Dialogue(dialogue_text)
        else:
            return Dialogue("That's all the info I have for ye.")

    def render(self, surface):
        # Draw NPC as a colored circle
        pygame.draw.circle(surface, GREEN, self.position, 25)
        # Draw name above NPC
        name_text = FONT.render(self.name, True, WHITE)
        surface.blit(name_text, (self.position[0] - name_text.get_width() // 2, self.position[1] - 40))

class Player:
    def __init__(self, position=(100, 100)):
        self.rect = pygame.Rect(position, (PLAYER_SIZE, PLAYER_SIZE))
        self.color = GOLD
        self.animation_counter = 0
        self.direction = 'down'

    def move(self, dx, dy):
        self.rect.x += dx
        self.rect.y += dy

        # Keep player within screen bounds
        self.rect.x = max(0, min(self.rect.x, SCREEN_WIDTH - PLAYER_SIZE))
        self.rect.y = max(0, min(self.rect.y, SCREEN_HEIGHT - PLAYER_SIZE))

        # Update direction based on movement
        if dy < 0:
            self.direction = 'up'
        elif dy > 0:
            self.direction = 'down'
        elif dx < 0:
            self.direction = 'left'
        elif dx > 0:
            self.direction = 'right'

    def render(self, surface):
        # Simple animation by changing color
        if self.animation_counter < 30:
            color = self.color
        else:
            color = YELLOW
        pygame.draw.rect(surface, color, self.rect)
        self.animation_counter = (self.animation_counter + 1) % 60

class Scene:
    def __init__(self, name, background_color, npcs=None, puzzles=None):
        self.name = name
        self.background_color = background_color
        self.npcs = npcs if npcs else []
        self.puzzles = puzzles if puzzles else []
        self.objects = []  # For future use (items, interactive objects)

    def render_background(self, surface):
        surface.fill(self.background_color)

    def render_npcs(self, surface):
        for npc in self.npcs:
            npc.render(surface)

class Game:
    def __init__(self):
        # Load game state
        self.save_data = load_game()

        # Initialize player
        self.player = Player(position=(100, 100))

        # Initialize inventory
        self.inventory = Inventory(items=self.save_data.get("inventory", []))

        # Initialize puzzles
        self.puzzles = {
            "treasure": Puzzle(
                name="treasure",
                hint="I be thinkin' ye need to find the golden key!",
                answer="golden key",
                on_solve=self.unlock_treasure
            )
        }

        # Initialize NPCs
        self.npcs = [
            NPC("Captain Blackbeard", (800, 600), [
                "Ahoy, matey! Welcome aboard me ship!",
                "Beware of the hidden treasures on this ship."
            ]),
            NPC("Island Mystic", (200, 500), [
                "Arrr, the golden key be guarded by the spirits.",
                "Find the key to unlock the secrets of the island."
            ])
        ]

        # Initialize scenes
        self.scenes = {
            "ship": Scene("Ship", DARK_BLUE, npcs=[self.npcs[0]]),
            "island": Scene("Island", GREEN, npcs=[self.npcs[1]]),
            "tavern": Scene("Tavern", BROWN)
        }

        self.current_scene = self.scenes.get(self.save_data.get("current_scene", "ship"), self.scenes["ship"])

        # Dialogue management
        self.dialogue = None

        # Sound setup
        self.background_music = None
        self.play_music("ship")  # Start with ship music

        # Save flag
        self.needs_save = False

    def play_music(self, scene_name):
        # Placeholder: In a real game, load different music tracks per scene
        if self.background_music:
            self.background_music.stop()
        # For placeholder, we'll not load actual music files
        # Instead, you could use Pygame's built-in sounds or silence
        # self.background_music = pygame.mixer.Sound('path_to_music_file')
        # self.background_music.play(-1)
        pass  # No actual music

    def unlock_treasure(self):
        self.inventory.add_item("Golden Treasure")
        self.needs_save = True

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.save_game()
                pygame.quit()
                sys.exit()

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_i:
                    self.inventory.toggle()

                elif event.key == pygame.K_s:
                    self.save_game()
                    self.dialogue = Dialogue("Game Saved Successfully!")

                elif event.key == pygame.K_l:
                    self.load_game_state()
                    self.dialogue = Dialogue("Game Loaded Successfully!")

                elif event.key == pygame.K_SPACE:
                    # Attempt to solve puzzle
                    if not self.puzzles["treasure"].is_solved:
                        # Check if player has the required item
                        if self.inventory.has_item("Golden Key"):
                            success, message = self.puzzles["treasure"].attempt_solve("golden key")
                            self.dialogue = Dialogue(message)
                            self.needs_save = True
                        else:
                            self.dialogue = Dialogue("Ye don't have the golden key yet!")

                elif event.key == pygame.K_RETURN:
                    if self.dialogue:
                        self.dialogue = None

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    mouse_pos = event.pos
                    # Check NPC interactions
                    for npc in self.current_scene.npcs:
                        npc_rect = pygame.Rect(npc.position[0]-30, npc.position[1]-30, 60, 60)
                        if npc_rect.collidepoint(mouse_pos):
                            self.dialogue = npc.interact()
                            break
                    else:
                        # Check scene transitions based on clicked areas (placeholder)
                        if 900 < mouse_pos[0] < 950 and 700 < mouse_pos[1] < 750:
                            self.transition_scene("island")
                        elif 50 < mouse_pos[0] < 100 and 700 < mouse_pos[1] < 750:
                            self.transition_scene("tavern")

    def transition_scene(self, scene_name):
        if scene_name in self.scenes:
            self.current_scene = self.scenes[scene_name]
            self.play_music(scene_name)
            self.dialogue = Dialogue(f"Arrr! Welcome to the {scene_name.capitalize()}!")
            self.needs_save = True

    def update(self):
        keys = pygame.key.get_pressed()
        dx = dy = 0
        if keys[pygame.K_LEFT]:
            dx -= PLAYER_SPEED
        if keys[pygame.K_RIGHT]:
            dx += PLAYER_SPEED
        if keys[pygame.K_UP]:
            dy -= PLAYER_SPEED
        if keys[pygame.K_DOWN]:
            dy += PLAYER_SPEED
        self.player.move(dx, dy)

    def render_background(self, surface):
        self.current_scene.render_background(surface)

        # Draw scene-specific elements (placeholder)
        if self.current_scene.name == "ship":
            # Draw ship deck boundaries
            pygame.draw.rect(surface, BROWN, (0, 650, SCREEN_WIDTH, 100))
            # Draw transition areas
            pygame.draw.rect(surface, GRAY, (900, 700, 50, 50))  # Island
            pygame.draw.rect(surface, GRAY, (50, 700, 50, 50))    # Tavern
            # Labels
            label_island = FONT.render("Island", True, WHITE)
            surface.blit(label_island, (900, 760))
            label_tavern = FONT.render("Tavern", True, WHITE)
            surface.blit(label_tavern, (50, 760))

        elif self.current_scene.name == "island":
            # Draw island features
            pygame.draw.circle(surface, GREEN, (SCREEN_WIDTH//2, SCREEN_HEIGHT//2), 200)
            # Draw transition back to ship
            pygame.draw.rect(surface, GRAY, (500, 50, 50, 50))  # Back to Ship
            label_back = FONT.render("Ship", True, WHITE)
            surface.blit(label_back, (500, 110))

        elif self.current_scene.name == "tavern":
            # Draw tavern interior
            pygame.draw.rect(surface, BROWN, (0, 0, SCREEN_WIDTH, SCREEN_HEIGHT))
            # Draw transition back to ship
            pygame.draw.rect(surface, GRAY, (900, 700, 50, 50))  # Back to Ship
            label_back = FONT.render("Ship", True, WHITE)
            surface.blit(label_back, (900, 760))

    def render_npcs(self, surface):
        self.current_scene.render_npcs(surface)

    def render(self, surface):
        self.render_background(surface)
        self.player.render(surface)
        self.render_npcs(surface)
        if self.inventory.visible:
            self.inventory.render(surface)
        if self.dialogue:
            self.dialogue.render(surface)

    def save_game(self):
        self.save_data["current_scene"] = self.current_scene.name
        self.save_data["inventory"] = self.inventory.items
        self.save_data["puzzles_solved"] = {name: puzzle.is_solved for name, puzzle in self.puzzles.items()}
        save_game(self.save_data)

    def load_game_state(self):
        self.save_data = load_game()
        self.current_scene = self.scenes.get(self.save_data.get("current_scene", "ship"), self.scenes["ship"])
        self.inventory.items = self.save_data.get("inventory", [])
        # Load puzzles
        for name, solved in self.save_data.get("puzzles_solved", {}).items():
            if name in self.puzzles:
                self.puzzles[name].is_solved = solved
        self.play_music(self.current_scene.name)

    def run(self):
        while True:
            self.handle_events()
            self.update()
            self.render(screen)
            pygame.display.flip()
            clock.tick(FPS)

# Running the Game
if __name__ == "__main__":
    game = Game()
    game.run()
