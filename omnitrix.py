import pygame
import math
import os
import time

pygame.init()

WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Omnitrix Selector")

BLACK = (0, 0, 0)

# Load holograms from folder
icon_dir = "hologram_icons_glow"
icons_list = [pygame.image.load(os.path.join(icon_dir, f)).convert_alpha()
              for f in sorted(os.listdir(icon_dir)) if f.endswith(".png")]

# Ring settings
num_icons = len(icons_list)
radius = 200
rotation_speed = 0

# Load sounds
rotation_sound = pygame.mixer.Sound("rotation.wav") if os.path.exists("rotation.wav") else None
select_sound = pygame.mixer.Sound("select.wav") if os.path.exists("select.wav") else None

running = True
clock = pygame.time.Clock()

# Initial angles
angles = [(i / num_icons) * 2 * math.pi for i in range(num_icons)]

selected_index = None
selection_time = None
selection_duration = 0.5  # seconds

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if selected_index is None:  # allow rotation only if not selected
                if event.key == pygame.K_RIGHT:
                    rotation_speed = -0.05
                    if rotation_sound: rotation_sound.play()
                elif event.key == pygame.K_LEFT:
                    rotation_speed = 0.05
                    if rotation_sound: rotation_sound.play()
                elif event.key == pygame.K_RETURN:
                    # Select the one closest to top
                    top_angle = min(range(num_icons), key=lambda i: abs(math.sin(angles[i])))
                    selected_index = top_angle
                    selection_time = time.time()
                    if select_sound: select_sound.play()
        elif event.type == pygame.KEYUP:
            if event.key in [pygame.K_LEFT, pygame.K_RIGHT]:
                rotation_speed = 0

    if selected_index is None:
        # Update rotation if not in selection mode
        angles = [(a + rotation_speed) for a in angles]

    # Draw background
    screen.fill(BLACK)

    # Draw holograms
    for idx, icon in enumerate(icons_list):
        x = WIDTH // 2 + math.cos(angles[idx]) * radius
        y = HEIGHT // 2 + math.sin(angles[idx]) * radius

        draw_icon = icon
        if idx == selected_index:
            # Selection animation
            elapsed = time.time() - selection_time
            scale_factor = 1 + 0.5 * min(elapsed / selection_duration, 1)
            glow_intensity = min(elapsed / selection_duration, 1) * 255
            size = (int(icon.get_width() * scale_factor), int(icon.get_height() * scale_factor))
            draw_icon = pygame.transform.smoothscale(icon, size)

            # Optional glow
            glow_surface = pygame.Surface(size, pygame.SRCALPHA)
            pygame.draw.circle(glow_surface, (0, 255, 0, int(glow_intensity)), (size[0] // 2, size[1] // 2), size[0] // 2)
            screen.blit(glow_surface, (x - size[0] // 2, y - size[1] // 2))

        screen.blit(draw_icon, (x - draw_icon.get_width() // 2, y - draw_icon.get_height() // 2))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
