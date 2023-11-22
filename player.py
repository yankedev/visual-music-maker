import threading
import pygame
import os
import logging
from pydantic import BaseModel
from typing import List, Optional
import mimetypes
import time
import sys


class Colors:
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    RESET = "\033[0m"

    def colorize(text, color):
        return f"{color}{text}{Colors.RESET}"


class ColoredLevelNameFormatter(logging.Formatter):
    LEVEL_COLORS = {
        "WARNING": Colors.YELLOW,  # Yellow
        "INFO": Colors.BLUE,  # Blue
        "DEBUG": Colors.GREEN,  # Green
        "CRITICAL": Colors.RED,  # Red
        "ERROR": Colors.RED,  # Red
    }
    RESET_COLOR = "\033[0m"

    def format(self, record):
        levelname = record.levelname
        if levelname in self.LEVEL_COLORS:
            levelname_color = self.LEVEL_COLORS[levelname]
            record.levelname = f"{levelname_color}{levelname}{self.RESET_COLOR}"
        return super(ColoredLevelNameFormatter, self).format(record)


def audio_test():
    pygame.mixer.init()
    audio_file = (
        ".music/1-24. Droopy likes your face.flac"
    )
    song = pygame.mixer.Sound(audio_file)
    song.play()
    pygame.mixer.quit()


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
for handler in logging.root.handlers:
    handler.setFormatter(ColoredLevelNameFormatter("[%(levelname)s] %(message)s"))


class Song(BaseModel):
    file_path: str

    @staticmethod
    def from_file(file_path: str) -> Optional["Song"]:
        if not os.path.isfile(file_path):
            logging.warning(f"File does not exist: {file_path}")
            return None
        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type and mime_type.startswith("audio"):
            return Song(file_path=file_path)
        else:
            logging.warning(f"File is not a recognized audio type: {file_path}")
            return None


class Playlist(BaseModel):
    queue: List[Song] = []

    def is_empty(self):
        if len(self.queue) == 0:
            return True
        return False

    def add_song(self, song: Song) -> None:
        self.queue.append(song)
        logging.info(f"Song added: {song.file_path}")

    def remove_song(self, song: Song) -> None:
        self.queue.remove(song)
        logging.info(f"Song removed: {song.file_path}")

    def sort_songs(self) -> None:
        self.queue.sort(key=lambda song: os.path.basename(song.file_path))
        logging.info("Playlist sorted by song filenames.")

    @staticmethod
    def from_folder(folder_path: str, sort: bool = True) -> "Playlist":
        playlist = Playlist()
        if os.path.isdir(folder_path):
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                song = Song.from_file(file_path)
                if song:
                    playlist.add_song(song)
            playlist.sort_songs()
            return playlist
        else:
            logging.warning(f"Path '{folder_path}' is not a valid folder")
            return None

    def exchange_order(self, index1: int, index2: int) -> None:
        if index1 < len(self.queue) and index2 < len(self.queue):
            self.queue[index1], self.queue[index2] = (
                self.queue[index2],
                self.queue[index1],
            )
            logging.info(f"Exchanged songs at index {index1} and {index2}.")
        else:
            logging.warning("Invalid indices for exchanging songs.")

    def move_song(self, from_index: int, to_index: int) -> None:
        if from_index < len(self.queue) and to_index < len(self.queue):
            song = self.queue.pop(from_index)
            self.queue.insert(to_index, song)
            logging.info(f"Moved song from index {from_index} to {to_index}.")
        else:
            logging.warning("Invalid indices for moving song.")


class MusicPlayer:
    instance_count = 0

    def __init__(self, playlist=None):
        self.id = MusicPlayer.instance_count
        MusicPlayer.instance_count += 1

        logging.info(
            f"Initializing MusicPlayer with ID: {Colors.colorize(self.id, Colors.MAGENTA)}"
        )
        if not pygame.mixer.get_init():
            pygame.mixer.init()
        self.channel = pygame.mixer.Channel(self.instance_count)
        self.playlist = playlist if playlist else Playlist()
        self.is_playing = False
        self.current_song = None
        self.skip_requested = False
        self.playback_lock = threading.Lock()
        self.player_info = (
            f"[Player#{Colors.colorize(self.instance_count, Colors.MAGENTA)}] "
        )
        self.set_volume(1)
        self.set_index(0)

    def play(self):
        with self.playback_lock:
            if self.is_playing:
                logging.info(f"{self.player_info}is already playing.")
                return
            logging.info(f"{self.player_info}Starting playback...")
            self.is_playing = True
            threading.Thread(target=self._play_music).start()

    def _play_music(self):
        try:
            while self.is_playing and not self.playlist.is_empty():
                self.current_song = self.playlist.queue[self.current_index]
                logging.info(
                    f"{self.player_info}Now playing: {Colors.colorize(os.path.basename(self.current_song.file_path), Colors.YELLOW)}"
                )
                track = pygame.mixer.Sound(self.current_song.file_path)
                track.set_volume(self.volume)
                self.channel.play(track)
                while self.channel.get_busy():
                    pygame.time.delay(100)
                with self.playback_lock:
                    if self.is_playing and not self.skip_requested:
                        self._next_song()
        except Exception as e:
            logging.error(f"{self.player_info}Error during playback: {e}")
            sys.stdout.flush()

    def _next_song(self):
        self.current_index = (self.current_index + 1) % len(self.playlist.queue)
        logging.info(
            f"{self.player_info}Moving to song with index: {Colors.colorize(self.current_index, Colors.YELLOW)}"
        )

    def skip_song(self):
        with self.playback_lock:
            if self.is_playing and not self.playlist.is_empty():
                logging.info(f"{self.player_info}Skipping to next song...")
                self.skip_requested = True
                self._next_song()
                self._play_current_song()

    def previous_song(self):
        with self.playback_lock:
            if self.is_playing and not self.playlist.is_empty():
                logging.info(f"{self.player_info}Going back to previous song...")
                self.current_index = (self.current_index - 1) % len(self.playlist.queue)
                self._play_current_song()

    def _play_current_song(self):
        if not self.playlist.is_empty():
            self.channel.stop()
            self.current_song = self.playlist.queue[self.current_index]
            track = pygame.mixer.Sound(self.current_song.file_path)
            track.set_volume(self.volume)
            self.channel.play(track)

    def set_playlist(self, playlist):
        with self.playback_lock:
            self.playlist = playlist
            self.current_index = 0
            logging.info(f"{self.player_info}Playlist set.")

    def set_volume(self, volume):
        self.volume = volume
        self.left_volume = volume
        self.right_volume = volume
        if self.current_song:
            self.channel.set_volume(volume, volume)
            logging.info(
                f"{self.player_info}{Colors.colorize('Master', Colors.BLUE)} volume set to {Colors.colorize(volume, Colors.YELLOW)}"
            )

    def set_right_volume(self, volume):
        self.right_volume = volume
        with self.playback_lock:
            self.channel.set_volume(volume, self.left_volume)
            logging.info(
                f"{self.player_info}{Colors.colorize('Right', Colors.GREEN)} volume set to {Colors.colorize(volume, Colors.YELLOW)}"
            )

    def set_left_volume(self, volume):
        self.left_volume = volume
        with self.playback_lock:
            self.channel.set_volume(self.right_volume, volume)
            logging.info(
                f"{self.player_info}{Colors.colorize('Left', Colors.GREEN)} volume set to {Colors.colorize(volume, Colors.YELLOW)}"
            )

    def stop(self):
        with self.playback_lock:
            self.is_playing = False
            self.channel.stop()
            logging.info(f"{self.player_info}Playback stopped.")

    def set_index(self, index):
        with self.playback_lock:
            if 0 <= index < len(self.playlist.queue):
                self.current_index = index
                logging.info(
                    f"{self.player_info}Index set to: {Colors.colorize(index, Colors.YELLOW)}"
                )
            else:
                logging.warning(
                    f"{self.player_info}Invalid index: {Colors.colorize(index, Colors.RED)}"
                )

    def play_song(self, index):
        self.set_index(index)
        with self.playback_lock:
            if 0 <= index < len(self.playlist.queue):
                self._play_current_song()
            else:
                return

        if not self.is_playing:
            self.play()


playlist = Playlist.from_folder("./music")
if playlist and not playlist.is_empty():
    p1 = MusicPlayer(playlist)
    p2 = MusicPlayer(playlist)
    p1.set_index(3)
    p1.play()
    time.sleep(5)
    p1.play_song(1)
    # p1.skip_song()
    # p1.previous_song()
    time.sleep(10)
    p1.set_volume(1)
    time.sleep(5)
    p1.set_right_volume(0.2)
    time.sleep(5)
    p1.set_left_volume(0.2)
    p1.set_right_volume(1)
    time.sleep(5)
    p1.set_volume(1)
    time.sleep(5)
    p2.set_index(3)
    p2.play()
    time.sleep(10)
    p1.skip_song()
    p1.skip_song()
    p1.skip_song()
    time.sleep(10)
    p1.stop()
    time.sleep(3)
    p2.stop()
