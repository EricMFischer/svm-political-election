TRAITS = [
    ['old', 'not old'],
    ['masculine', 'not masculine'],
    ['baby-faced', 'not baby-faced'],
    ['competent', 'not competent'],
    ['attractive', 'not attractive'],
    ['energetic', 'not energetic'],
    ['well-groomed', 'not well-groomed'],
    ['intelligent', 'not intelligent'],
    ['honest', 'not honest'],
    ['generous', 'not generous'],
    ['trustworthy', 'not trustworthy'],
    ['confident', 'not confident'],
    ['rich', 'not rich'],
    ['dominant', 'not dominant']
]

# param_grid = [{ # v3
    #     'kernel': ['rbf'],
    #     'C': [2**-3, 2**-2, 2**-1, 2**1, 2**2, 2**3, 2**4, 2**5, 2**6, 2**7, 2**8, 2**9, 2**10, 2**11, 2**12],
    #     'gamma': [2**-9, 2**-8, 2**-7, 2**-6, 2**-5, 2**-4, 2**-3, 2**-2, 2**-1, 2**1, 2**2, 2**3, 2**4, 2**5],
    #     'epsilon': [2**-8, 2**-7, 2**-6, 2**-5, 2**-4, 2**-3, 2**-2, 2**-1]
    # }]

'''
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
ax1.axis('off')
ax1.imshow(img, cmap=plt.cm.gray)
ax1.set_title('Input image')
# Rescale histogram for better display
hog_img_rescaled = exposure.rescale_intensity(hog_img, in_range=(0, 10))
ax2.axis('off')
ax2.imshow(hog_img_rescaled, cmap=plt.cm.gray)
ax2.set_title('Histogram of Oriented Gradients')
plt.show()
'''
