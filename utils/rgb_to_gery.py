from PIL import Image
import os.path as osp
import os
import glob
# dir='/home/lyq/Desktop/dataset/Reid/SYSU-MM'
# camlist = [1,2,4,5]
# for i in camlist:
#     campath = dir + '/cam' + str(i)
#     for root, dirs, files in os.walk(campath, topdown=False):
#         for name in dirs:
#             img_paths = glob.glob(osp.join(campath, name, '*.jpg'))
#             save_dir = dir + '/cam'+str(i)+'_gery/'+name
#             if not osp.exists(save_dir):
#                 os.mkdir(save_dir)
#             for path in img_paths:
#                 img = Image.open(path).convert('LA')
#                 img.save(save_dir+'/'+path[-8:-4]+'.png')
#     print(campath+' finish.')


# dir='/home/lyq/Desktop/dataset/Reid/RegDB/Visible/'
# for root, dirs, files in os.walk(dir, topdown=False):
#     for name in dirs:
#         img_paths = glob.glob(osp.join(dir, name, '*.bmp'))
#         save_dir = dir+'_gery/'+name
#         if not osp.exists(save_dir):
#             os.mkdir(save_dir)
#         for path in img_paths:
#             img = Image.open(path).convert('LA')
#             img.save(save_dir+'/'+path.split('/')[-1][:-4]+'.png')
# print(' finish.')

# dir='/home/lyq/Desktop/dataset/Reid/Market/bounding_box_train'
# save_dir = dir + '_gery/'
# if not osp.exists(save_dir):
#     os.mkdir(save_dir)
# img_paths = glob.glob(osp.join(dir, '*.jpg'))
# for path in img_paths:
#     img = Image.open(path).convert('LA')
#     img.save(save_dir+'/'+path.split('/')[-1][:-4]+'.png')
# print(' finish.')

dir='/home/lyq/Desktop/dataset/Reid/Market/query'
save_dir = dir + '_gery/'
if not osp.exists(save_dir):
    os.mkdir(save_dir)
img_paths = glob.glob(osp.join(dir, '*.jpg'))
for path in img_paths:
    img = Image.open(path).convert('LA')
    img.save(save_dir+'/'+path.split('/')[-1][:-4]+'.png')
print(' finish.')