import json
import pandas as pd
import numpy as np
import urllib.error
import urllib.request
import shutil
import os



def compute_distance( lat0, lng0, lat1, lng1, unit='km' ):
    """
    data format : JGD2011 (GRS80)
    formula     : https://www.trail-note.net/tech/calc_distance/
    """
    scale = 1.0
    if unit in ['m','M']:
        scale = 1.0
    if unit in ['km','kM','Km','KM']:
        scale = 1000.0
    
    Rx = 6378137.0	
    Ry = 6356752.314140
    #E  = np.sqrt( 1 - (Ry/Rx)**2 )
    E2  = 1 - (Ry/Rx)**2
    Dy = ( lat0 - lat1 ) * ( np.pi / 180.0 )
    Dx = ( lng0 - lng1 ) * ( np.pi / 180.0 )
    P  = ( lat0 + lat1 ) * ( np.pi / 180.0 * 0.5 )
    W  = np.sqrt( 1 - E2 * np.square( np.sin(P) ) )
    M  = Rx * ( 1 - E2 ) / np.power( W, 3 )
    N  = Rx / W
    D  = np.sqrt( np.square(Dy*M) + np.square(Dx*N*np.cos(P)) )

    return D / scale



def gen_data_ascsv( year=2020, dp='data/station', download='not always' ):
    """
    Only FY2020 data can be properly processed
    """
    fp_csv1 = f'{dp}/station_list.csv'
    fp_csv2 = f'{dp}/passengers.csv'

    fp_json = download_data( year=year, dp=dp, always=(download=='always') )
    df1,df2 = load_data_asdf( fp_json, year )
    
    df1.to_csv( fp_csv1, index=False )
    df2.to_csv( fp_csv2, index=False )



def download_data( year=2020, dp='data/passengers', always=False ):
    # Path
    YY      = f'{year%100:02}'    
    fn_zip  = f'S12-{YY}_GML.zip'
    fn_json = f'S12-{YY}_NumberOfPassengers.geojson'
    url     = f'https://nlftp.mlit.go.jp/ksj/gml/data/S12/S12-{YY}/{fn_zip}'
    dp_zip  = f'{dp}/{os.path.splitext(fn_zip)[0]}'
    fp_zip  = f'{dp}/{fn_zip}'
    fp_src  = f'{dp_zip}/{fn_json}'
    fp_dst  = f'{dp}/{fn_json}'

    # Check file existance
    if os.path.exists( fp_dst ) and ( not always ):
        print('[Info] Skip downloading')
        return fp_dst
    else:
        print('[Info] Start downloading')

    # Download    
    try:
        with urllib.request.urlopen( url ) as file_download:
            data = file_download.read()
            with open( fp_zip, mode='wb' ) as file_save:
                file_save.write( data )
            file_save.close()
    except urllib.error.URLError as e:
        print(e)
   
    # Extract -> Copy -> Clean
    shutil.unpack_archive( fp_zip, dp )
    shutil.copyfile( fp_src, fp_dst )
    shutil.rmtree( dp_zip )
    os.remove( fp_zip )

    return fp_dst



def load_data_asdf( fp, year=2020 ):
    # Check file existance
    if not os.path.exists( fp ):
        print(f'[Error] file does not exist: {fp}')
        return pd.DataFrame()
    else:
        print('[Info] Start processing')

    # Conversion
    lut    = gen_lookuptable( year )
    d_json = load_data_asjson( fp )
    d_list = []
    valids = []
    for i in range(len(d_json)):
        coordinates = d_json[i]['geometry']['coordinates'][0]
        d = {
            'lat0': f'{coordinates[0][1]:.6f}',
            'lng0': f'{coordinates[0][0]:.6f}',
            'lat1': f'{coordinates[1][1]:.6f}',
            'lng1': f'{coordinates[1][0]:.6f}',
        }
        for k, v in lut['name'].items():
            d[v] = str( d_json[i]['properties'][k] )
        for k, v in lut['data'].items():
            d[v] = str( d_json[i]['properties'][k] )
        d_list.append( d )
    for i in range(len(d_json)):
        invalid = True
        for k, v in lut['meta'].items():
            # 2: count is integrated to another station
            # 3: station does not exist
            invalid &= ( d_json[i]['properties'][k] in [2,3] )
        valids.append( (not invalid) )

    # Post process    
    df  = pd.DataFrame( d_list )
    df = df.sort_values( ['lat0','lng0'], ascending=[False,False] )
    df = df.reset_index( drop=True )
    df1 = df.loc[:,['lat0','lng0','lat1','lng1','station_name','company_name','line_name']]
    df2 = df.iloc[valids].replace( '0', np.nan ).reset_index( drop=True )
    df3 = remove_duplicate( df2 )
    print( len(df1),len(df2), len(df3) )

    #return df1, df2
    return df1, df3



def load_data_asjson( fp ):
    with open( fp, 'r', encoding='utf-8_sig' ) as f:
        d = json.load(f)
    f.close()
    return d['features']



def gen_lookuptable( year=2020 ):
    lut_name = {
        'S12_001': 'station_name',
        'S12_002': 'company_name',
        'S12_003': 'line_name',
    }
    lut_data = {}
    for y in range(2011,year):
        ix  = 9 + 4 * ( y - 2011 )
        key = f'S12_{ix:03}'
        lut_data[key] = str( y )
    lut_meta = {}
    for y in range(2011,year):
        ix  = 6 + 4 * ( y - 2011 )
        key = f'S12_{ix:03}'
        lut_meta[key] = str( y )
    lut = {
        'name': lut_name,
        'data': lut_data,
        'meta': lut_meta,
    }
    return lut



def remove_duplicate( df_in, thrshd=0.2 ):
    # Target
    cols   = ['lat0','lng0','lat1','lng1','station_name','company_name','line_name']
    empty  = df_in.drop( columns=cols ).isna().all( axis=1 )
    dupl   = df_in['station_name'].duplicated(keep=False)
    #ix_cnt = ( empty & dupl ) & ( ~ (~empty)&(dupl) )
    ix_cnt = ( empty & dupl )

    # Distance
    n      = len(df_in)
    lats   = df_in['lat0'].astype(float).values.reshape(n,1)
    lngs   = df_in['lng0'].astype(float).values.reshape(n,1)
    dists  = compute_distance( lats, lngs, lats, lngs )        
    mask   = ~ np.eye(n).astype(bool)
    ix_dst = ( ( dists < thrshd ) & mask ).any( axis=1 )

    # Valid
    ix_vld = ~ ( ix_cnt & ix_dst )

    return df_in[ix_vld]



def test():
    lat0 = [ 36.10377477777778, 30]
    lng0 = [140.08785502777778,125]
    lat1 = [ 35.65502847222223, 25]
    lng1 = [139.74475044444443,130]
    lat0 = np.array( lat0 )
    lng0 = np.array( lng0 )
    lat1 = np.array( lat1 )
    lng1 = np.array( lng1 )
    d = compute_distance( lat0, lng0, lat1, lng1 )
    print(d)



if __name__ == '__main__':
    gen_data_ascsv( year=2020 )    
    #test()