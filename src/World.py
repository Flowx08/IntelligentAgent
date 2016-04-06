#Import modules
import pygame

#Virtual world
class virtual_world:

	def __init__(self, width, height):
		self.width = width
		self.height = height
		self.data = [[0 for i in xrange(self.width)] for k in xrange(self.height)]
		self.step = 0
	
	#Place object in the world
	def place_object(self, obj, x, y):
		self.data[x % self.width][y % self.height] = obj
	
	def get_object(self, x, y):
		return self.data[x % self.width][y % self.height]

	def nextstep(self):
		self.step += 1

	#draw world matrix
	def draw(self, screen):
		for x in xrange(self.width):
			for y in xrange(self.height):
				if self.data[x][y] == 0: pygame.draw.rect(screen, (100, 100, 100), (x*16, y*16, 16, 16))
				elif self.data[x][y] == 1: pygame.draw.rect(screen, (100, 255, 100), (x*16, y*16, 16, 16))
				elif self.data[x][y] == 2: pygame.draw.rect(screen, (255, 100, 100), (x*16, y*16, 16, 16))
				elif self.data[x][y] == 3: pygame.draw.rect(screen, (100, 100, 255), (x*16, y*16, 16, 16))
				elif self.data[x][y] == 4: pygame.draw.rect(screen, (255, 255, 100), (x*16, y*16, 16, 16))
