import asyncio
import random
import sys

import pygame
from pygame.locals import K_ESCAPE, K_SPACE, K_UP, KEYDOWN, QUIT

from .ai.game_observation import GameObservation
from .ai.game_result import GameResult
from .ai.model import GameAction, Model
from .entities import (
    Background,
    Floor,
    GameOver,
    Pipes,
    Player,
    PlayerMode,
    Score,
    WelcomeMessage,
)
from .utils import GameConfig, Images, Sounds, Window

NUMBER_OF_BIRDS = 10

class Flappy:
    def __init__(self):
        """
        Initializes the Flappy Bird game.

        Parameters:
            None

        Returns:
            None
        """
        pygame.init()
        pygame.display.set_caption("Flappy Bird")
        window = Window(288, 512)
        screen = pygame.display.set_mode((window.width, window.height))
        images = Images()

        self.config = GameConfig(
            screen=screen,
            clock=pygame.time.Clock(),
            fps=30,
            window=window,
            images=images,
            sounds=Sounds(),
        )

        self.human_player = (
            False if len(sys.argv) > 1 and sys.argv[1] == "ai" else True
        )
        if not self.human_player:
            self.models = [Model() for _ in range(NUMBER_OF_BIRDS)]
            self.model_results = []


    async def start(self):
        while True:
            self.background = Background(self.config)
            self.floor = Floor(self.config)

            self.players = [Player(self.config, random_y=random.randint(-10, 10)) for _ in range(NUMBER_OF_BIRDS)]

            self.welcome_message = WelcomeMessage(self.config)
            self.game_over_message = GameOver(self.config)
            self.pipes = Pipes(self.config)
            self.score = Score(self.config)

            if not self.human_player:
                print("Starting in AI mode")
                await self.agent_play_v2()
                # await self.game_over()
            else:
                await self.splash()
                await self.play()
                await self.game_over()

    async def splash(self):
        """Shows welcome splash screen animation of flappy bird"""

        for player in self.players:
            player.set_mode(PlayerMode.SHM)

        while True:
            for event in pygame.event.get():
                self.check_quit_event(event)
                if self.is_tap_event(event):
                    return

            self.background.tick()
            self.floor.tick()
            for player in self.players:
                player.tick()
            self.welcome_message.tick()

            pygame.display.update()
            await asyncio.sleep(0)
            self.config.tick()

    def check_quit_event(self, event):
        if event.type == QUIT or (
            event.type == KEYDOWN and event.key == K_ESCAPE
        ):
            pygame.quit()
            sys.exit()

    def is_tap_event(self, event):
        m_left, _, _ = pygame.mouse.get_pressed()
        space_or_up = event.type == KEYDOWN and (
            event.key == K_SPACE or event.key == K_UP
        )
        screen_tap = event.type == pygame.FINGERDOWN
        return m_left or space_or_up or screen_tap

    async def agent_play_v2(self):
        self.score.reset()

        for player in self.players:
            player.set_mode(PlayerMode.NORMAL)
        
        while True:
            for player, model in zip(self.players, self.models):
                # the observation we will pass to the AI agent
                observation = GameObservation(
                    bird_y_pos=player.y,
                    y_dist_to_bot_pipe=self.pipes.upper[0].y - player.y,
                    y_dist_to_top_pipe=self.pipes.lower[0].y - player.y,
                    x_dist_to_pipe_pair=self.pipes.upper[0].x -player.x,
                    bird_y_vel=player.vel_y,
                )

                # Get agent decision
                action = model.predict(observation)

                # Perform action
                if action == GameAction.JUMP and len(pygame.event.get()) == 0:
                    jump_event = pygame.event.Event(
                        pygame.KEYDOWN, {"key": pygame.K_SPACE}
                    )
                    pygame.event.post(jump_event)
                elif (
                    action == GameAction.DO_NOTHING and len(pygame.event.get()) > 0
                ):
                    pygame.event.clear()

                if player.collided(self.pipes, self.floor):
                    if(len(self.players) == 1):
                        return
                    self.players.remove(player)

                for i, pipe in enumerate(self.pipes.upper):
                    if player.crossed(pipe):
                        self.score.add()

                for event in pygame.event.get():
                    self.check_quit_event(event)
                    if self.is_tap_event(event):
                        player.flap()

                self.background.tick()
                self.floor.tick()
                self.pipes.tick()
                self.score.tick()
                for player in self.players:
                    player.tick()

                pygame.display.update()
                await asyncio.sleep(0)
                self.config.tick()
            
         
    async def agent_play(self):
        self.score.reset()
        self.player.set_mode(PlayerMode.NORMAL)

        while True:
            # the observation we will pass to the AI agent
            observation = GameObservation(
                bird_y_pos=self.player.y,
                y_dist_to_bot_pipe=self.pipes.upper[0].y - self.player.y,
                y_dist_to_top_pipe=self.pipes.lower[0].y - self.player.y,
                x_dist_to_pipe_pair=self.pipes.upper[0].x - self.player.x,
                bird_y_vel=self.player.vel_y,
            )

            # Get agent decision
            action = self.model.predict(observation)

            # Perform action
            if action == GameAction.JUMP and len(pygame.event.get()) == 0:
                jump_event = pygame.event.Event(
                    pygame.KEYDOWN, {"key": pygame.K_SPACE}
                )
                pygame.event.post(jump_event)
            elif (
                action == GameAction.DO_NOTHING and len(pygame.event.get()) > 0
            ):
                pygame.event.clear()

            if self.player.collided(self.pipes, self.floor):
                return

            for i, pipe in enumerate(self.pipes.upper):
                if self.player.crossed(pipe):
                    self.score.add()

            for event in pygame.event.get():
                self.check_quit_event(event)
                if self.is_tap_event(event):
                    self.player.flap()

            self.background.tick()
            self.floor.tick()
            self.pipes.tick()
            self.score.tick()
            self.player.tick()

            pygame.display.update()
            await asyncio.sleep(0)
            self.config.tick()
    
    async def play(self):
        self.score.reset()
        self.player.set_mode(PlayerMode.NORMAL)

        while True:
            if self.player.collided(self.pipes, self.floor):
                return

            for i, pipe in enumerate(self.pipes.upper):
                if self.player.crossed(pipe):
                    self.score.add()

            for event in pygame.event.get():
                self.check_quit_event(event)
                if self.is_tap_event(event):
                    self.player.flap()

            self.background.tick()
            self.floor.tick()
            self.pipes.tick()
            self.score.tick()
            self.player.tick()

            pygame.display.update()
            await asyncio.sleep(0)
            self.config.tick()

    async def game_over(self):
        """crashes the player down and shows gameover image"""

        self.player.set_mode(PlayerMode.CRASH)
        self.pipes.stop()
        self.floor.stop()

        if self.human_player:
            while True:
                for event in pygame.event.get():
                    self.check_quit_event(event)
                    if self.is_tap_event(event):
                        if self.player.y + self.player.h >= self.floor.y - 1:
                            return

                self.background.tick()
                self.floor.tick()
                self.pipes.tick()
                self.score.tick()
                self.player.tick()
                self.game_over_message.tick()

                self.config.tick()
                pygame.display.update()
                await asyncio.sleep(0)
        else:
            # AI player
            print("AI agent lost. Restarting...")
            self.model_results.append(GameResult(self.score.score))

            self.background.tick()
            self.floor.tick()
            self.pipes.tick()
            self.score.tick()
            self.player.tick()
            self.game_over_message.tick()

            self.config.tick()
            pygame.display.update()
            await asyncio.sleep(0)
