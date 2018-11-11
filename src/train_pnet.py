import net
import train

if __name__ == '__main__':
    ne = net.PNet()
    ne.train()
    trainer = train.Trainer(ne, 'E:/save_path/12/param/',r"E:\save_path\12")
    trainer.train()