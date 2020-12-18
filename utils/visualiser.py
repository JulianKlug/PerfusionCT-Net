import numpy as np
import pandas as pd
from collections import OrderedDict
import os
import ntpath
import time
from utils import utils, html
from utils.plot_logs import plot_logs

# Use the following comment to launch a visdom server
# python -m visdom.server

class Visualiser():
    def __init__(self, opt, save_dir, filename='loss_log.txt'):
        self.display_id = opt.display_id
        self.use_html = not opt.no_html
        self.win_size = opt.display_winsize
        self.save_epoch_freq = opt.save_epoch_freq
        # if save epoch frequency option is negative, volumes will not be saved
        self.save_to_disk = True if opt.save_epoch_freq >= 0 else False
        self.save_dir = save_dir
        self.name = os.path.basename(self.save_dir)
        self.saved = False
        self.display_single_pane_ncols = opt.display_single_pane_ncols

        # Error plots
        self.error_plots = dict()
        self.error_wins = dict()

        if self.display_id > 0:
            import visdom
            self.vis = visdom.Visdom(port=opt.display_port)

        if self.save_volumes:
            self.saved_volumes_dir = os.path.join(self.save_dir, 'saved_volumes')
            utils.mkdir(self.saved_volumes_dir)

        if self.use_html:
            self.web_dir = os.path.join(self.save_dir, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            print('create web directory %s...' % self.web_dir)
            utils.mkdirs([self.web_dir, self.img_dir])
        self.log_name = os.path.join(self.save_dir, filename)
        self.log_table = os.path.join(self.save_dir, filename.split('.')[0] + '.xlsx')
        if os.path.exists(self.log_table):
            timestamp = str(time.time()).split('.')[0]
            self.log_table = os.path.splitext(self.log_table)[0] + '_' + timestamp + '.xlsx'

        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def reset(self):
        self.saved = False

    def display_current_volumes(self, volumes, ids, split, epoch):
        if not self.display_id > 0 or not self.save_to_disk:  # iamges not needed
            return
        volumes = OrderedDict([('output', utils.volume2img(volumes['output'])),
                              ('input', utils.volume2img(volumes['input'])),
                              ('target', utils.volume2img(volumes['target']))])
        if self.save_to_disk and epoch % self.save_epoch_freq == 0:
            self.save_volumes(volumes, ids, split, epoch)
        if not self.display_id > 0:  # don't show images in the browser
            return
        for i in range(len(ids)):
            volume_stack = np.concatenate((volumes['input'][i], volumes['output'][i, 0:1], volumes['target'][i]), axis=0)
            volume_stack = np.expand_dims(volume_stack, axis=1)
            volume_stack = np.transpose(volume_stack, (0, 1, 3, 2))
            volume_stack = np.nan_to_num(volume_stack)
            self.vis.images(volume_stack, opts=dict(title=f'Prediction: {split} {epoch} {ids[i]}'),
                            win=self.display_id + i)

    # |visuals|: dictionary of images to display or save
    def display_current_results(self, visuals, epoch, save_result, ids):
        if self.display_id > 0:  # show images in the browser
            ncols = self.display_single_pane_ncols
            if ncols > 0:
                h, w = next(iter(visuals.values())).shape[:2]
                table_css = """<style>
                        table {border-collapse: separate; border-spacing:4px; white-space:nowrap; text-align:center}
                        table td {width: %dpx; height: %dpx; padding: 4px; outline: 4px solid black}
                        </style>""" % (w, h)
                title = self.name
                label_html = ''
                label_html_row = ''
                nrows = int(np.ceil(len(visuals.items()) / ncols))
                images = []
                idx = 0
                for label, image_numpy in visuals.items():
                    label_html_row += '<td>%s</td>' % label
                    images.append(image_numpy.transpose([2, 0, 1]))
                    idx += 1
                    if idx % ncols == 0:
                        label_html += '<tr>%s</tr>' % label_html_row
                        label_html_row = ''
                white_image = np.ones_like(image_numpy.transpose([2, 0, 1]))*255
                while idx % ncols != 0:
                    images.append(white_image)
                    label_html_row += '<td></td>'
                    idx += 1
                if label_html_row != '':
                    label_html += '<tr>%s</tr>' % label_html_row
                # pane col = image row
                self.vis.images(images, nrow=ncols, win=self.display_id + 1,
                                padding=2, opts=dict(title=title + ' images'))
                label_html = '<table>%s</table>' % label_html
                self.vis.text(table_css + label_html, win=self.display_id + 2,
                              opts=dict(title=title + ' labels'))
            else:
                idx = 1
                for label, image_numpy in visuals.items():
                    self.vis.image(image_numpy.transpose([2, 0, 1]), opts=dict(title=ids + '_' + label),
                                   win=self.display_id + idx)
                    idx += 1

        if self.use_html and (save_result or not self.saved):  # save images to a html file
            self.saved = True
            for label, image_numpy in visuals.items():
                img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.png' % (epoch, label))
                utils.save_image(image_numpy, img_path)
            # update website
            webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, reflesh=1)
            for n in range(epoch, 0, -1):
                webpage.add_header('epoch [%d]' % n)
                ims = []
                txts = []
                links = []

                for label, image_numpy in visuals.items():
                    img_path = 'epoch%.3d_%s.png' % (n, label)
                    ims.append(img_path)
                    txts.append(label)
                    links.append(img_path)
                webpage.add_images(ims, txts, links, width=self.win_size)
            webpage.save()

    def plot_table_html(self, x, y, key, split_name, **kwargs):
        key_s = key+'_'+split_name
        if key_s not in self.error_plots:
            self.error_wins[key_s] = self.display_id * 3 + len(self.error_wins)
        else:
            self.vis.close(self.error_plots[key_s])


        table = pd.DataFrame(np.array(y['data']).transpose(),
                             index=kwargs['labels'], columns=y['colnames'])
        table_html = table.round(2).to_html(col_space=200, bold_rows=True, border=12)

        self.error_plots[key_s] = self.vis.text(table_html,
                                                opts=dict(title=self.name+split_name,
                                                          width=350, height=350,
                                                          win=self.error_wins[key_s]))


    def plot_heatmap(self, x, y, key, split_name, **kwargs):
        key_s = key+'_'+split_name
        if key_s not in self.error_plots:
            self.error_wins[key_s] = self.display_id * 3 + len(self.error_wins)
        else:
            self.vis.close(self.error_plots[key_s])
        self.error_plots[key_s] = self.vis.heatmap(
            X=y,
            opts=dict(
                columnnames=kwargs['labels'],
                rownames=kwargs['labels'],
                title=self.name + ' confusion matrix',
                win=self.error_wins[key_s]))

    def plot_line(self, x, y, key, split_name):
        if key not in self.error_plots:
            self.error_wins[key] = self.display_id * 3 + len(self.error_wins)
            self.error_plots[key] = self.vis.line(
                X=np.array([x, x]),
                Y=np.array([y, y]),
                opts=dict(
                    legend=[split_name],
                    title=self.name + ' {} over time'.format(key),
                    xlabel='Epochs',
                    ylabel=key,
                    win=self.error_wins[key]
            ))
        else:
            self.vis.line(X=np.array([x]), Y=np.array([y]), win=self.error_plots[key], name=split_name, update='append')
    # errors: dictionary of error labels and values
    def plot_current_errors(self, epoch, errors, split_name, counter_ratio=0.0, **kwargs):
        if self.display_id > 0:
            for key in errors.keys():
                x = epoch + counter_ratio
                y = errors[key]
                if isinstance(y, dict):
                    if y['type'] == 'table':
                        self.plot_table_html(x,y,key,split_name, **kwargs)
                elif np.isscalar(y):
                    self.plot_line(x,y,key,split_name)
                elif y.ndim == 2:
                    self.plot_heatmap(x,y,key,split_name, **kwargs)


    # errors: same format as |errors| of plotCurrentErrors
    def print_current_errors(self, epoch, errors, split_name):
        message = '(epoch: %d, split: %s) ' % (epoch, split_name)
        error_df = [epoch]
        error_df_header = ['epoch']
        for k, v in errors.items():
            if np.isscalar(v):
                message += '%s: %.3f ' % (k, v)
                error_df.append(v)
                error_df_header.append(k)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

        error_df = pd.DataFrame(data=[error_df], columns=error_df_header)
        if epoch == 0:
            utils.append_df_to_excel(self.log_table, error_df, sheet_name=split_name, header=True)
        else:
            utils.append_df_to_excel(self.log_table, error_df, sheet_name=split_name, header=False)

    def save_plots(self, epoch, save_frequency):
        if epoch % save_frequency == 0:
            plot_logs(self.log_table)

    def save_volumes(self, volumes, ids, split, epoch):
        epoch_dir = os.path.join(self.saved_volumes_dir, f'epoch_{str(epoch)}')
        split_dir = os.path.join(epoch_dir, split)
        utils.mkdirs([epoch_dir, split_dir])
        utils.save_volumes(volumes, ids, split_dir)

    # save image to the disk
    def save_images(self, webpage, visuals, image_path):
        image_dir = webpage.get_image_dir()
        short_path = ntpath.basename(image_path[0])
        name = os.path.splitext(short_path)[0]

        webpage.add_header(name)
        ims = []
        txts = []
        links = []

        for label, image_numpy in visuals.items():
            image_name = '%s_%s.png' % (name, label)
            save_path = os.path.join(image_dir, image_name)
            utils.save_image(image_numpy, save_path)

            ims.append(image_name)
            txts.append(label)
            links.append(image_name)
        webpage.add_images(ims, txts, links, width=self.win_size)
