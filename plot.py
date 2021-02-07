c2 = steel_prediction(raw_train)
c2['ClassId'] = c2['ImageId_ClassId'].apply(lambda x:int(x[-1]))
c2['ImageId'] = c2['ImageId_ClassId'].apply(lambda x:x.split('/')[-1][:-2])
c2['Area'] = c2['EncodedPixels'].apply(lambda x: sum(list(map(int,x.split(' ')))[1::2]) if x else 0)
c2
u = pd.DataFrame({'ImageId': train_image_names})
# u['hasDefect'] = (c3['ImageId']==u['ImageId'])&(c3['hasDefect']>0.9)
u['predDefect'] = np.isin(u['ImageId'], c3['ImageId'])
u['trueDefect'] = np.isin(u['ImageId'], trainLabels['ImageId'])
u = u.merge(c3, how='outer', on=['ImageId'])
u['predClass'] = u['ClassId'].fillna(0).astype('int')
u.drop('ClassId', inplace=True, axis=1)
# u['predClass'] = c3[['defect_1','defect_2','defect_3','defect_4']].idxmax(axis=1).apply(lambda x:int(x[-1]))
# u.iloc[np.isin(u['ImageId'], trainLabels['ImageId'])]['trueClass'] = trainLabels['ClassId']
# u = u.combine_first(trainLabels)
# u = u.merge(trainLabels, on=['ImageId'])
# u.fillna(0, inplace=True)
u = u.merge(trainLabels, how='outer', on=['ImageId'])
u['trueClass'] = u['ClassId'].fillna(0).astype('int')
u