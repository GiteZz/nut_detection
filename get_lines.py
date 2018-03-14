import cv2
import numpy as np
from matplotlib import pyplot as plt
import bolt_const
import math
import logging
import time

lines = []

def check_point(co_y, co_x, dist):
    clear_plot()
    debug = False
    best_line = None
    plot_point([co_y, co_x])

    # check square surrounding coordinate, if point found use co and point to get a rough angle
    square = give_square_co(co_y, co_x, 4)
    for co in square:
        plot_point(co, color='r')
        if within_bounds(co):
            if img[co[0]][co[1]]:
                plot_point(co, color='b')
                print(co)
                line_info = check_line(co_y, co_x, co[0], co[1])
                print("found new line with hitrate: " + str(line_info['hitrate']))
                if best_line is not None:
                    if line_info['hitrate'] > best_line['hitrate']:
                        best_line = line_info
                else:
                    best_line = line_info

    if best_line is not None:
        if best_line['hitrate'] > bolt_const.line_treshold_first:
            clear_plot()
            best_line = get_better_line(best_line)
            print("best hitrate is: " + str(best_line['hitrate']))
            clear_plot()
            if best_line['hitrate'] > bolt_const.line_treshold_second:
                lines.append(best_line)
                remove_area_line(best_line)
            #b_plot_line(best_line)
            #plot_hitpoints(best_line)

        else:
            img[co_y][co_x] = 0
    else:
        img[co_y][co_x] = 0


def remove_area_line(line):
    debug = False
    print("started clearing image")
    global im
    vec = line['vector']
    start_point_index, end_point_index, length = get_min_max_line(line)
    co_start_x = line['hitpoints_x'][start_point_index]
    co_start_y = line['hitpoints_y'][start_point_index]

    for i in range(bolt_const.miss_start, int(length) - bolt_const.miss_start +1):
        p_x = co_start_x + vec[1]*i
        p_y = co_start_y + vec[0]*i
        for b in range(0, int(length*bolt_const.remove_width) + 1):
            for x in range(2):
                for y in range(2):
                    p1_x = int(p_x + vec[0]*b) + x
                    p1_y = int(p_y - vec[1]*b) + y
                    if within_bounds((p1_y,p1_x)):
                        img[p1_y][p1_x] = 0

                    p2_x = int(p_x - vec[0] * b) + x
                    p2_y = int(p_y + vec[1] * b) + y
                    if within_bounds((p2_y, p2_x)):
                        img[p2_y][p2_x] = 0

                    im.set_data(img)
                    if debug:
                        plot_point((p2_y, p2_x))
                        plot_point((p1_y, p1_x))
        if debug:
            plot_point((p_y, p_x))


def clear_plot():
    global plot_elem
    for el in plot_elem:
        el.remove()
    plot_elem = []


def b_plot_line(line):
    start_point_index, end_point_index, length = get_min_max_line(line)
    hitpoints_x = line['hitpoints_x']
    hitpoints_y = line['hitpoints_y']
    ln, = plt.plot((hitpoints_x[start_point_index], hitpoints_x[end_point_index]),
             (hitpoints_y[start_point_index], hitpoints_y[end_point_index]))
    plot_elem.append(ln)
    plt.pause(bolt_const.graph_update_time)


def get_better_line(line, iteration=0, sensitivity = 0.05):
    clear_plot()
    hitpoints_x = line['hitpoints_x']
    hitpoints_y = line['hitpoints_y']

    start_point_index, end_point_index, length = get_min_max_line(line)

    #plt.plot((hitpoints_x[start_point_index], hitpoints_x[end_point_index]), (hitpoints_y[start_point_index], hitpoints_y[end_point_index]))
    extra_rot = [1-sensitivity,1,1+sensitivity]
    best_line = line
    for i in range(0, len(extra_rot)):
        if len(hitpoints_x) > 1:
            a, b = np.polyfit(hitpoints_x, hitpoints_y, 1)
            a *= extra_rot[i]
            vec = np.array([1, a])
        else:
            vec = line['vector']

        norm = np.linalg.norm(vec)
        mult = length/norm
        vec = vec*mult
        end_point_x = hitpoints_x[start_point_index] - vec[0]*1.5
        end_point_y = hitpoints_y[start_point_index] - vec[1]*1.5
        # plt.plot((hitpoints_x[start_point_index], end_point_x),
        #          (hitpoints_y[start_point_index], end_point_y))
        new_line = check_line(hitpoints_y[start_point_index], hitpoints_x[start_point_index], end_point_y, end_point_x)

        if new_line['hitrate'] >= best_line['hitrate']:

            print("improved line from: " + str(line['hitrate']) + " to line with: " + str(new_line['hitrate']) + " in iteration: " + str(iteration))
            best_line = new_line
        else:
            print("degraded line from: " + str(line['hitrate']) + " to line with: " + str(new_line['hitrate']) + " in iteration: " + str(iteration))

    print("in iteration: " + str(iteration) + " we changed from " + str(line['hitrate']) + " to: " + str(best_line['hitrate']))
    #b_plot_line(best_line)
    if best_line['hitrate'] > line['hitrate']:
        if best_line['hitrate'] - line['hitrate'] > bolt_const.change_sensitivity:
            return get_better_line(best_line, iteration=iteration+1, sensitivity=sensitivity)
        else:
            return get_better_line(best_line, iteration=iteration + 1, sensitivity=sensitivity/2)
    else:
        return line


#function gives the index of the 2 most outest points from the hitpoint in line
def get_min_max_line(line):
    hitpoints_x = line['hitpoints_x']
    hitpoints_y = line['hitpoints_y']

    max_x = max(hitpoints_x)
    min_x = min(hitpoints_x)
    max_y = max(hitpoints_y)
    min_y = min(hitpoints_y)

    if abs(max_x - min_x) > abs(max_y - min_y):
        start_point_index = hitpoints_x.index(max_x)
        end_point_index = hitpoints_x.index(min_x)
    else:
        start_point_index = hitpoints_y.index(max_y)
        end_point_index = hitpoints_y.index(min_y)

    length = math.sqrt(math.pow(hitpoints_x[start_point_index] - hitpoints_x[end_point_index], 2) + math.pow(
        hitpoints_y[start_point_index] - hitpoints_y[end_point_index], 2))

    return start_point_index, end_point_index, length


def check_line(co1_y, co1_x, co2_y, co2_x,length = bolt_const.edge_length_cst_mod):
    debug = True
    vec = np.array([co2_y-co1_y, co2_x-co1_x])

    norm = np.linalg.norm(vec)
    vec = vec / norm
    hitpoints_x = []
    hitpoints_y = []
    dist = math.sqrt(length/bolt_const.div_segments)
    hits = 0
    for i in range(-bolt_const.div_segments,bolt_const.div_segments+1):
        sub_hits = 0
        sub_hitpoints_x = []
        sub_hitpoints_y = []
        for x in range(-1,3):
            for y in range(-1,3):
                #Get rounded up and rounded down for x and y, 4 comb
                p_x = int(co1_x + vec[1]*dist*i) + x
                p_y = int(co1_y + vec[0]*dist*i) + y
                if within_bounds((p_y, p_x)):
                    if img[p_y, p_x]:
                        hits += 1
                        sub_hits += 1
                        sub_hitpoints_x.append(p_x)
                        sub_hitpoints_y.append(p_y)
                        if debug:
                            plot_point([p_y,p_x],color='g')
                    else:
                        if debug:
                            plot_point([p_y, p_x], color='w')
        if sub_hits > 0:
            hitpoints_x.append(sum(sub_hitpoints_x) / float(sub_hits))
            hitpoints_y.append(sum(sub_hitpoints_y) / float(sub_hits))

    hitrate = hits / (bolt_const.div_segments*2*4.0)
    if debug:
        ln, = plt.plot([co1_x + vec[1]*dist*bolt_const.div_segments, co1_x - vec[1]*dist*bolt_const.div_segments],
                       [co1_y + vec[0]*dist*bolt_const.div_segments, co1_y - vec[0]*dist*bolt_const.div_segments])
        plot_elem.append(ln)
    return {'hitrate': hitrate, 'co': (co1_y, co1_x), 'vector': vec, 'hitpoints_x': hitpoints_x, 'hitpoints_y': hitpoints_y}


def within_bounds(co):
    co_y = co[0]
    co_x = co[1]
    return 0 <= co_x < img_width and 0 <= co_y < img_height


def plot_hitpoints(line):
    for i in range(len(line['hitpoints_x'])):
        plot_point((line['hitpoints_y'][i],line['hitpoints_x'][i]), color='r')


def give_square_co(y_co,x_co,radius):
    co_list = []
    for x in range(x_co - radius, x_co + radius + 1):
        co_list.append([y_co + radius, x])
        co_list.append([y_co - radius, x])

    for y in range(y_co - radius + 1, y_co + radius):
        co_list.append([y, x_co + radius])
        co_list.append([y, x_co - radius])

    return co_list


def plot_point(co, color='r'):
    debug = True
    # opencv use y, x instead of x, y
    if debug:
        pt, = plt.plot(co[1], co[0], color + 'o', markersize=1.5)
        plot_elem.append(pt)
        plt.pause(bolt_const.graph_update_time)


def prep():
    x_max = [0] * img_height
    x_min = [img_width - 1] * img_height

    y_max = [0] * img_width
    y_min = [img_height - 1] * img_width

    for y in range(0, img_height):
        for x in range(0, img_width):
            if img[y][x]:
                if x_max[y] < x:
                    x_max[y] = x
                if x_min[y] > x:
                    x_min[y] = x

                if y_max[x] < y:
                    y_max[x] = y
                if y_min[x] > y:
                    y_min[x] = y

    for y in range(0, img_height):
        for x in range(0, img_width):
            if img[y][x]:
                if x_min[y] < x < x_max[y]:
                    img[y][x] = 0
                if y_min[x] < y < y_max[x]:
                    img[y][x] = 0
                im.set_data(img)

            # if y_max[x] == 0 and x > 1:
            #     y_max[x] = y_max[x - 1]
            #     img[y_max[x - 1]][x] = 1
            #     im.set_data(img)
            #
            # if y_min[x] == img_height and x > 1:
            #     y_min[x] = y_min[x - 1]
            #     img[y_min[x - 1]][x] = 1
            #     im.set_data(img)
    print("ended prep")





img = cv2.imread('canny/1.png',0)
print(len(img))
img_height = len(img)
img_width = len(img[0])

plot_elem = []

im = plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])

plt.ion()
stop_loop = False
prep()
for y in range(0, img_height):
    for x in range(0, img_width):
        if img[y][x]:
            print(img[y][x])
            check_point(y,x, 3)
            stop_loop = False
            if stop_loop:
                break
    if stop_loop:
        break

for l in lines:
    b_plot_line(l)

plt.pause(5000)

plt.show()