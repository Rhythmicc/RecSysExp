import sys
sys.path.append("../")
try:
    import B
    from B.DataImporter import *
    from B.KFoldCheck import cross_validation
    from B.Models import NCF, MF, AE
    from time import perf_counter
except ImportError:
    exit("FUCK")

print("Starting ml-100k\n-----------")

if '--enable-baseline' in sys.argv:
    print("MF Model")
    ml_data_df = import_movieLens_100k_data()
    B.L2, B.EPOCH = 1e-4, 100
    s = perf_counter()
    cross_validation(ml_data_df, MF.model)
    e = perf_counter()
    print("Use time: %.4f\n" % (e-s))

print("AutoEncoder Model")
ml_data_df = import_movieLens_100k_data()
B.L2, B.EPOCH = 1e-4, 10
s = perf_counter()
cross_validation(ml_data_df, AE.model)
e = perf_counter()
print("Use time: %.4f\n" % (e-s))

print("Starting amazon\n-----------")

if '--enable-baseline' in sys.argv:
    print("MF Model")
    amazon_data_df = import_amazon_data()
    B.L2, B.EPOCH = 1e-5, 150
    s = perf_counter()
    cross_validation(amazon_data_df, MF.model)
    e = perf_counter()
    print("Use time: %.4f\n" % (e-s))

print("NCF Model")
amazon_data_df = import_amazon_data()
B.L2, B.EPOCH = 1e-5, 15
s = perf_counter()
cross_validation(amazon_data_df, NCF.model)
e = perf_counter()
print("Use time: %.4f\n" % (e-s))

