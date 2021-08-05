import numpy as np
import h5py
import sys
import os
import cv2
from itertools import islice
from scipy import ndimage
import logging

log = logging.getLogger(__name__)

def chunk(iterable, size):
    """Splitting by chunks"""
    it = iter(iterable)
    item = list(islice(it, size))
    while item:
        yield np.array(item)
        item = list(islice(it, size))

def apply_shift(data, p):    
    """Apply (p[0],p[1]) shift of data"""
    res = data.copy()
    for k in range(data.shape[0]):
        res[k] = ndimage.shift(data[k],p[k],mode='nearest',order=1)
    return res

def find_min_max(flat):
    """Find min and max values according to histogram"""
    h, e = np.histogram(flat[:], 1000)
    stend = np.where(h > np.max(h)*0.005)
    st = stend[0][0]
    end = stend[0][-1]
    mmin = e[st]
    mmax = e[end+1]
    return mmin, mmax

def register_shift_sift(data, flat):
    """Find shifts via SIFT detecting features"""
    mmin, mmax = find_min_max(flat)
    sift = cv2.SIFT_create()
    shifts = np.zeros([data.shape[0], 2], dtype='float32')
    for id in range(data.shape[0]):
        tmp1 = ((data[id]-mmin) /
                (mmax-mmin)*255)
        tmp1[tmp1 > 255] = 255
        tmp1[tmp1 < 0] = 0
        tmp2 = ((flat-mmin) /
                (mmax-mmin)*255)
        tmp2[tmp2 > 255] = 255
        tmp2[tmp2 < 0] = 0
        # find key points
        tmp1 = tmp1.astype('uint8')
        tmp2 = tmp2.astype('uint8')

        kp1, des1 = sift.detectAndCompute(tmp1, None)
        kp2, des2 = sift.detectAndCompute(tmp2, None)
        # cv2.imwrite('original_image_right_keypoints.png',
        #             cv2.drawKeypoints(tmp1, kp1, None))
        # cv2.imwrite('original_image_left_keypoints.png',
        #             cv2.drawKeypoints(tmp2, kp2, None))
        match = cv2.BFMatcher()
        matches = match.knnMatch(des1, des2, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.5*n.distance:
                good.append(m)

        draw_params = dict(matchColor=(0, 255, 0),
                           singlePointColor=None,
                           flags=2)
        tmp3 = cv2.drawMatches(tmp1, kp1, tmp2, kp2, good, None, **draw_params)
        # cv2.imwrite("original_image_drawMatches.jpg", tmp3)
        src_pts = np.float32(
            [kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32(
            [kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        shift = (src_pts-dst_pts)[:, 0, :]
        shifts[id] = np.median(shift, axis=0)[::-1]
        #print(
            #f'number of matched points for data {id}: {len(good)}, found shifts:{shifts[id]}')
        if(len(good)==0):
            log.warning(f'no feature matches, set shift to 0')
            shifts[id] = 0

    return shifts

def flat_drift_correction(params):
    """
    Fix drift of flat field during data acquistion 
    by using a small region not containing the sample.
    Note: the method may not work if the region contains a part of the sample
    """ 
    file_name = params.file_name
    xs = params.flat_region_startx
    xe = params.flat_region_endx
    ys = params.flat_region_starty
    ye = params.flat_region_endy
    proj_chunk = params.nproj_per_chunk
    average_shift = params.average_shift_per_chunk
    log.info(f'file name {file_name}')
    log.info(f'flat region x:({xs}-{xe}), y:({ys}-{ye})')
    log.info(f'average shift per chunk:{average_shift}')
    
    file_out_name = str(file_name)[:-3]+'_corr.h5'
    log.info(f'create a new h5 file {file_out_name}')  
    
    with  h5py.File(file_name, 'r') as file_fid:
        with  h5py.File(file_out_name, 'w') as file_out_fid:
    
            data = file_fid['exchange/data']
            flat = file_fid['exchange/data_white'][:].astype('float32')
            dark = file_fid['exchange/data_dark'][:].astype('float32')

            for a in file_fid.attrs:
                file_out_fid.attrs[a] = file_fid.attrs[a]
            for d in file_fid:
                if 'exchange' == d:
                    file_out_fid.create_group('exchange')
                    for dd in file_fid[d]:
                        if 'data' != dd:
                            file_fid.copy('exchange/'+dd,file_out_fid['exchange'])
                else:
                    file_fid.copy(d,file_out_fid)
            data_corr = file_out_fid.create_dataset("/exchange/data", data.shape,
                                            chunks=(1, data.shape[1], data.shape[2]), dtype='float')
            flat_corr = file_out_fid['exchange/data_white']
            dark_corr = file_out_fid['exchange/data_dark']
            flat_corr[:] = 1
            dark_corr[:] = 0
            
            log.info(f'register flat fields')
            dark_median = np.median(dark, axis=0)
            dark_median_part = dark_median[ys:ye, xs:xe] 
            flat_part = flat[:, ys:ye, xs:xe]   
            shifts = register_shift_sift(flat_part-dark_median_part,np.median(flat_part-dark_median_part, axis=0))
            flat_shift = apply_shift(flat, -shifts)  
            flat_shift_median = np.median(flat_shift, axis=0)
            flat_shift_median_part = flat_shift_median[ys:ye, xs:xe]            
            
            for ids in chunk(range(data.shape[0]),proj_chunk):
                # find flat field shifts w.r.t. each projection by using small parts without sample
                log.info(f'processing projections {ids[0]}-{ids[-1]}')
                # read data part
                data_part = data[ids, ys:ye, xs:xe][:].astype('float32')                          
                # register shifts             
                shifts = register_shift_sift(data_part-dark_median_part, flat_shift_median_part-dark_median_part)
                if(average_shift):
                    ashift = np.median(shifts,axis=0)
                    log.info(f'average shift  {ashift}')
                    shifts[:] = ashift
                # read chunk of projections                   
                data_chunk = data[ids][:].astype('float32')
                # apply shifts'                       
                flat_shift_median_shift = apply_shift(np.tile(flat_shift_median,[data_chunk.shape[0],1,1]), shifts)
                dark_median_shift = apply_shift(np.tile(dark_median,[data_chunk.shape[0],1,1]), shifts)        
                # apply flat field correction                           
                data_corr_chunk = (data_chunk-dark_median)/(flat_shift_median_shift-dark_median_shift+1e-5)
                data_corr[ids] = data_corr_chunk
            log.info(f'data is saved to {file_out_name}')
                
                
                
