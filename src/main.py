#Import modules
import random
import IntelligentAgent as ai
import World
import pygame
import sys

#Game
pygame.init()

#Screen informations
width = 800
height = 600
screen = pygame.display.set_mode((width, height))
turbo = False
world = virtual_world(32, 32)
foodspawn = 0
foodfreq = 3
spikespawn = 0
spikefreq = 3
agent = ai.IntelligentAgent(2048, 0.01, 0.8, 0.9) 
agent_x = random.randint(0, world.width-1)
agent_y = random.randint(0, world.height-1)
reward = 0.0
cumulativereward = 0.0
mediumreward = 0.0
totalreward = 0.0
world.place_object(3, agent_x, agent_y)

#Font for text
font = pygame.font.SysFont("Monaco", 18)

def update_world():
	global foodspawn
	global spikespawn
	global world
	foodspawn -= 1
	spikespawn -= 1
	if foodspawn <= 0:
		x = random.randint(0, world.width-1)
		y = random.randint(0, world.height-1)
		j = 0
		while world.data[x][y] == 3 and j < 10:
			j += 1
			x = random.randint(0, world.width-1)
			y = random.randint(0, world.height-1)
		world.place_object(1, x, y)
		foodspawn = foodfreq
	
	if spikespawn <= 0:
		x = random.randint(0, world.width-1)
		y = random.randint(0, world.height-1)
		j = 0
		while world.data[x][y] == 3 and j < 10:
			j += 1
			x = random.randint(0, world.width-1)
			y = random.randint(0, world.height-1)
		world.place_object(2, x, y)
		spikespawn = spikefreq

def update_agent():
	global agent
	global agent_x
	global agent_y
	global world
	global reward
	global mediumreward
	global cumulativereward
	global totalreward

	#Select action
	state = []
	for x in range(agent_x-3, agent_x+4):
		for y in range(agent_y-3, agent_y+4):
			if world.data[x % world.width][y % world.height] == 1: state.append(1.0)
			elif world.data[x % world.width][y % world.height] == 2: state.append(-1.0)
			else: state.append(0.0)
	agent.learn(state, reward)
	action = agent.step(state)

	#Action and reward
	if action == 0:
		obj = world.get_object(agent_x-1, agent_y)
		if obj == 0: reward = 0.0
		elif obj == 1: reward = 1.0
		elif obj == 2: reward = -1.0
		world.place_object(0, agent_x, agent_y)
		world.place_object(3, agent_x-1, agent_y)
		agent_x -= 1
		if agent_x < 0: agent_x += world.width	
	elif action == 1:
		obj = world.get_object(agent_x+1, agent_y)
		if obj == 0: reward = 0.0
		elif obj == 1: reward = 1.0
		elif obj == 2: reward = -1.0
		world.place_object(0, agent_x, agent_y)
		world.place_object(3, agent_x+1, agent_y)
		agent_x += 1
		if agent_x >= world.width: agent_x = 0	
	elif action == 2:
		obj = world.get_object(agent_x, agent_y-1)
		if obj == 0: reward = 0.0
		elif obj == 1: reward = 1.0
		elif obj == 2: reward = -1.0
		world.place_object(0, agent_x, agent_y)
		world.place_object(3, agent_x, agent_y-1)
		agent_y -= 1
		if agent_y < 0: agent_y += 32	
	elif action == 3:
		obj = world.get_object(agent_x, agent_y+1)
		if obj == 0: reward = 0.0
		elif obj == 1: reward = 1.0
		elif obj == 2: reward = -1.0
		world.place_object(0, agent_x, agent_y)
		world.place_object(3, agent_x, agent_y+1)
		agent_y += 1
		if agent_y >= world.height: agent_y = 0
	
	#Medium reward
	cumulativereward += reward
	totalreward += reward
	if world.step % 1000 == 0:
		mediumreward = cumulativereward / 1000.0
		agent.save()
		cumulativereward = 0



#Draw text at position x,y
def draw_text(text, x, y):
	label = font.render(text, 1, (255, 255, 255))
	screen.blit(label, (x, y))

#Game loop
while 1:
	global turbo
	
	#Increment world step
	world.nextstep()

	#Update virtual world with food and spikes
	update_world()
	update_agent()

	#Event loop
	for event in pygame.event.get():
		if event.type == pygame.QUIT: sys.exit()
		elif event.type == pygame.KEYDOWN:
			if event.key == pygame.K_s: turbo = True
			if event.key == pygame.K_o: agent.curiosity = (agent.curiosity - 0.05) % 1.0
			if event.key == pygame.K_p: agent.curiosity = (agent.curiosity + 0.05) % 1.0
		elif event.type == pygame.KEYUP:
			if event.key == pygame.K_s: turbo = False
	if turbo: continue

	#Clear black screen
	screen.fill((0, 0, 0))

	#Draw the virtual world
	world.draw()

	#Draw info
	draw_text("Informations:", world.width*16 + 16, 16*0 + 1)
	draw_text("Step: {}".format(world.step), world.width*16 + 16, 16*1 + 1)
	draw_text("Agent X:{} Y:{}".format(agent_x, agent_y), world.width*16 + 16, 16*2 + 1)
	draw_text("Curiosity: {}".format(agent.curiosity), world.width*16 + 16, 16*3 + 1)
	draw_text("Reward: {}".format(reward), world.width*16 + 16, 16*4 + 1)
	draw_text("MediumReward: {}".format(mediumreward), world.width*16 + 16, 16*5 + 1)
	draw_text("TotalReward: {}".format(totalreward), world.width*16 + 16, 16*6 + 1)
	
	#Display everything
	pygame.display.flip()
	
	#60 FPS
	pygame.time.Clock().tick(60)

