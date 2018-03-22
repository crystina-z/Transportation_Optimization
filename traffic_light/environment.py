from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import optparse
import subprocess
import random

# import the traci library beforehead
try:
    sys.path.append(os.path.join(os.path.dirname(
        __file__), '..', '..', '..', '..', "tools"))
    sys.path.append(os.path.join(os.environ.get("SUMO_HOME", os.path.join(
        os.path.dirname(__file__), "..", "..", "..")), "tools"))
    from sumolib import checkBinary
except ImportError:
    sys.exit(
        "please declare environment variable 'SUMO_HOME' as the root directory of your sumo installation (it should contain folders 'bin', 'tools' and 'docs')")
import traci

class gameEnv():
    def __init__(self):
        self.step = 0
        self.actions = 4
        self.last_action = -1
        self.route_choice = {"EL":0,"ES":1,"ER":2,
                "SL":3,"SS":4,"SR":5,
                "WL":6,"WS":7,"WR":8,
                "NL":9,"NS":10,"NR":11
                }

        self.sum = 0.

    # generate the route file stored in data/tl.rou.xml
    def generate_routefile(self):
        random.seed(42)
        N = 3000
        with open("data/tl.rou.xml", "w") as routes:
            print("""<routes>
            <vType id="0" accel="0.8" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="16.67" guiShape="passenger"/>
            <route id="EL" edges="EI SO" />
            <route id="ES" edges="EI WO" />
            <route id="ER" edges="EI NO" />
            <route id="WL" edges="WI NO" />
            <route id="WS" edges="WI EO" />
            <route id="WR" edges="WI SO" />
            <route id="NL" edges="NI EO" />
            <route id="NS" edges="NI SO" />
            <route id="NR" edges="NI WO" />
            <route id="SL" edges="SI WO" />
            <route id="SS" edges="SI NO" />
            <route id="SR" edges="SI EO" />
            """, file=routes)
            lastVeh = 0
            vehNr = 0
            choice = {0:"EL",1:"ES",2:"ER",
            3:"SL",4:"SS",5:"SR",
            6:"WL",7:"WS",8:"WR",
            9:"NL",10:"NS",11:"NR"
            }
            p = [None for i in range(12)]
            for i in range(12):
                p[i] = random.random() * 0.1
            for i in range(N):
                for j in range(12):
                    r = random.random()
                    if r*r < p[j]:
                        print('    <vehicle id="%i" type="0" route="%s" depart="%i" />' % (
                            lastVeh,choice.get(j,), i), file=routes)
                        vehNr += 1
                        lastVeh += 1
            print("</routes>", file=routes)

    # determine the parameter of simulation
    def get_options(self):
        optParser = optparse.OptionParser()
        optParser.add_option("--nogui", action="store_true",
                             default=False, help="run the commandline version of sumo")
        options, args = optParser.parse_args()
        return options

    def actions(self):
        return self.actions

    def getAction(self, a_num):
        # a_num: 0 1 2 3
        # transform input number into phaseID in TraCI:
        phase = 0
        if 0 == a_num:
            phase = 0
        elif 1 == a_num:
            phase = 2
        elif 2 == a_num:
            phase = 4
        elif 3 == a_num:
            phase = 6
        return phase

    def getState(self):
        idlist = traci.vehicle.getIDList()
        poslist= [[0 for x in range(10)] for y in range(12)]

        for id in idlist:
            route = self.route_choice.get(traci.vehicle.getRouteID(id),)
            distance = traci.vehicle.getDistance(id)

            if distance < 500. and distance > -1.:
                poslist[route][int(distance/50)] += 1
        return poslist

    # initialize the environment and start a new episode of simulation
    def reset(self):
        options = self.get_options()

        sumoBinary = checkBinary('sumo')
        # sumoBinary = checkBinary('sumo-gui')

        traci.start([sumoBinary, "-c", "data/tl.sumocfg","--no-warnings"])

        # initialize environment
        traci.trafficlight.setPhase("C", 0)
        self.actions = 2
        self.step = 0
        self.sum = 0.
        return self.getState()
        #run()


    def getReward(self):
        idlist = traci.vehicle.getIDList()
        count = 0
        square_waiting_time = 0.
        factor = 0.
        for id in idlist:
            distance = traci.vehicle.getDistance(id)
            if distance > -1. and distance < 500.0:
                count += 1
                square_waiting_time += traci.vehicle.getWaitingTime(id)*traci.vehicle.getWaitingTime(id)
                factor += traci.vehicle.getSpeedFactor(id)
        if count == 0:
            return 0
        else:
            return (- 0.02 * square_waiting_time + factor)/count

    # simulation by one step(1 second)
    def stepForward(self, a):
        done = 0
        if(a == self.last_action):
            # traci.trafficlight.setPhase("C", a*2)
            traci.simulationStep()
            self.sum += self.getReward()
            self.step += 1

            # state = self.getState()
            # reward = self.getReward()
            done = (traci.simulation.getMinExpectedNumber() <= 0)

            # return state, reward, done
        else:
            for i in range(20):
                traci.trafficlight.setPhase("C", a*2)  # phase = a * 2
                traci.simulationStep()
                self.sum += self.getReward()
                self.step += 1

                if (traci.simulation.getMinExpectedNumber() <= 0):
                    done = 1
                    break

                self.last_action = a

        state = self.getState()
        reward = self.getReward()  # reward calculation method should be changed (average)
        return state, reward, done

    # terminate the current simulation
    def close(self):
        traci.close()
        sys.stdout.flush()
