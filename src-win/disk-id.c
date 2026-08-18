#include "plat.h"

#include <windows.h>
#include <cfgmgr32.h>
#include <setupapi.h>
#include <winioctl.h>

#define AIMDO_CACHE_MISS   (-2)
#define AIMDO_DISK_UNKNOWN (-1)
#define AIMDO_DISK_SLOW      0
#define AIMDO_DISK_FAST      1

#define BUS_TYPE_UNKNOWN             0
#define BUS_TYPE_USB                 7
#define BUS_TYPE_RAID                8
#define BUS_TYPE_ISCSI               9
#define BUS_TYPE_VIRTUAL            14
#define BUS_TYPE_FILE_BACKED_VIRTUAL 15
#define BUS_TYPE_SPACES             16
#define BUS_TYPE_NVME               17
#define BUS_TYPE_NVMEOF             20
#define BUS_TYPE_MAX                21

typedef struct DiskTopologyCache {
    struct DiskTopologyCache *next;
    wchar_t volume[64];
    DWORD disk_count;
    int result;
    DWORD disks[1];
} DiskTopologyCache;

static const GUID disk_interface_guid = {
    0x53f56307, 0xb6bf, 0x11d0, {0x94, 0xf2, 0x00, 0xa0, 0xc9, 0x1e, 0xfb, 0x8b}
};
static const GUID pci_property_guid = {
    0x3ab22e31, 0x8264, 0x4b4e, {0x9a, 0xf5, 0xa8, 0xd2, 0xd8, 0xe3, 0x3e, 0x62}
};
static SRWLOCK cache_lock = SRWLOCK_INIT;
static DiskTopologyCache *cache_entries;

static HANDLE open_existing(const wchar_t *path) {
    return CreateFileW(path, 0, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                       NULL, OPEN_EXISTING, 0, NULL);
}

static bool get_volume_guid(const wchar_t *path, wchar_t volume[64]) {
    wchar_t buffer[32768];
    wchar_t *end;
    DWORD length;
    HANDLE file = open_existing(path);

    if (file == INVALID_HANDLE_VALUE) {
        return false;
    }
    length = GetFinalPathNameByHandleW(file, buffer, ARRAY_SIZE(buffer),
                                       FILE_NAME_NORMALIZED | VOLUME_NAME_GUID);
    CloseHandle(file);
    if (!length || length >= ARRAY_SIZE(buffer) || wcsncmp(buffer, L"\\\\?\\Volume{", 11) ||
        !(end = wcschr(buffer + 11, L'}')) || end - buffer + 2 >= 64) {
        return false;
    }
    end[1] = L'\\';
    end[2] = L'\0';
    wcscpy_s(volume, 64, buffer);
    return true;
}

static int compare_dword(const void *a, const void *b) {
    DWORD left = *(const DWORD *)a;
    DWORD right = *(const DWORD *)b;
    return left > right ? 1 : left < right ? -1 : 0;
}

static bool get_volume_disks(const wchar_t *volume, DWORD **disks_out, DWORD *count_out) {
    VOLUME_DISK_EXTENTS *extents = NULL;
    DWORD *disks = NULL;
    DWORD bytes = 1024;
    DWORD returned;
    DWORD count;
    DWORD unique;
    wchar_t volume_path[64];
    HANDLE handle;
    bool success = false;

    wcscpy_s(volume_path, ARRAY_SIZE(volume_path), volume);
    volume_path[wcslen(volume_path) - 1] = L'\0';
    handle = open_existing(volume_path);
    if (handle == INVALID_HANDLE_VALUE) {
        return false;
    }
    while (bytes <= M) {
        VOLUME_DISK_EXTENTS *resized = realloc(extents, bytes);

        if (!resized) {
            goto done;
        }
        extents = resized;
        if (DeviceIoControl(handle, IOCTL_VOLUME_GET_VOLUME_DISK_EXTENTS,
                            NULL, 0, extents, bytes, &returned, NULL)) {
            break;
        }
        DWORD error = GetLastError();
        if (error != ERROR_MORE_DATA && error != ERROR_INSUFFICIENT_BUFFER) {
            goto done;
        }
        bytes *= 2;
    }
    if (bytes > M || returned < offsetof(VOLUME_DISK_EXTENTS, Extents)) {
        goto done;
    }
    count = extents->NumberOfDiskExtents;
    if (!count || count > (returned - offsetof(VOLUME_DISK_EXTENTS, Extents)) /
                          sizeof(extents->Extents[0])) {
        goto done;
    }
    disks = malloc(count * sizeof(*disks));
    if (!disks) {
        goto done;
    }
    for (DWORD i = 0; i < count; i++) {
        disks[i] = extents->Extents[i].DiskNumber;
    }
    qsort(disks, count, sizeof(*disks), compare_dword);
    unique = 1;
    for (DWORD i = 1; i < count; i++) {
        if (disks[i] != disks[unique - 1]) {
            disks[unique++] = disks[i];
        }
    }
    *disks_out = disks;
    *count_out = unique;
    disks = NULL;
    success = true;

done:
    free(disks);
    free(extents);
    CloseHandle(handle);
    return success;
}

static int cache_lookup(const wchar_t *volume, const DWORD *disks, DWORD count) {
    int result = AIMDO_CACHE_MISS;

    AcquireSRWLockShared(&cache_lock);
    for (DiskTopologyCache *entry = cache_entries; entry; entry = entry->next) {
        if (entry->disk_count == count && !_wcsicmp(entry->volume, volume) &&
            !memcmp(entry->disks, disks, count * sizeof(*disks))) {
            result = entry->result;
            break;
        }
    }
    ReleaseSRWLockShared(&cache_lock);
    return result;
}

static void cache_store(const wchar_t *volume, const DWORD *disks, DWORD count, int result) {
    DiskTopologyCache *entry = malloc(sizeof(*entry) + (count - 1) * sizeof(*disks));

    if (!entry) {
        return;
    }
    wcscpy_s(entry->volume, ARRAY_SIZE(entry->volume), volume);
    entry->disk_count = count;
    entry->result = result;
    memcpy(entry->disks, disks, count * sizeof(*disks));

    AcquireSRWLockExclusive(&cache_lock);
    entry->next = cache_entries;
    cache_entries = entry;
    ReleaseSRWLockExclusive(&cache_lock);
}

static int disk_bus_type(DWORD disk_number) {
    STORAGE_PROPERTY_QUERY query = {
        .PropertyId = StorageDeviceProperty,
        .QueryType = PropertyStandardQuery,
    };
    STORAGE_DEVICE_DESCRIPTOR descriptor;
    wchar_t path[64];
    DWORD returned;
    HANDLE disk;

    swprintf_s(path, ARRAY_SIZE(path), L"\\\\.\\PhysicalDrive%lu", disk_number);
    disk = open_existing(path);
    if (disk == INVALID_HANDLE_VALUE) {
        return -1;
    }
    if (!DeviceIoControl(disk, IOCTL_STORAGE_QUERY_PROPERTY,
                         &query, sizeof(query), &descriptor, sizeof(descriptor),
                         &returned, NULL) || returned < offsetof(STORAGE_DEVICE_DESCRIPTOR, RawDeviceProperties)) {
        CloseHandle(disk);
        return -1;
    }
    CloseHandle(disk);
    return descriptor.BusType;
}

static DEVINST disk_devinst(DWORD disk_number) {
    SP_DEVICE_INTERFACE_DATA interface_data = {0};
    SP_DEVINFO_DATA devinfo = {0};
    HDEVINFO devices;
    DEVINST found = 0;

    devices = SetupDiGetClassDevsW(&disk_interface_guid, NULL, NULL,
                                   DIGCF_PRESENT | DIGCF_DEVICEINTERFACE);
    if (devices == INVALID_HANDLE_VALUE) {
        return 0;
    }
    interface_data.cbSize = sizeof(interface_data);
    for (DWORD index = 0; SetupDiEnumDeviceInterfaces(
             devices, NULL, &disk_interface_guid, index, &interface_data); index++) {
        PSP_DEVICE_INTERFACE_DETAIL_DATA_W detail;
        STORAGE_DEVICE_NUMBER number;
        DWORD required = 0;
        DWORD returned;
        HANDLE disk;

        SetupDiGetDeviceInterfaceDetailW(devices, &interface_data, NULL, 0, &required, NULL);
        if (required < sizeof(*detail) || !(detail = malloc(required))) {
            continue;
        }
        detail->cbSize = sizeof(*detail);
        devinfo.cbSize = sizeof(devinfo);
        if (!SetupDiGetDeviceInterfaceDetailW(devices, &interface_data, detail, required,
                                              NULL, &devinfo) ||
            (disk = open_existing(detail->DevicePath)) == INVALID_HANDLE_VALUE) {
            free(detail);
            continue;
        }
        if (DeviceIoControl(disk, IOCTL_STORAGE_GET_DEVICE_NUMBER, NULL, 0,
                            &number, sizeof(number), &returned, NULL) &&
            returned >= sizeof(number) && number.DeviceType == FILE_DEVICE_DISK &&
            number.DeviceNumber == disk_number) {
            found = devinfo.DevInst;
        }
        CloseHandle(disk);
        free(detail);
        if (found) {
            break;
        }
    }
    SetupDiDestroyDeviceInfoList(devices);
    return found;
}

static bool pci_property(DEVINST devinst, DWORD pid, DWORD *value) {
    DEVPROPKEY key = {.fmtid = {0}, .pid = pid};
    DEVPROPTYPE type;
    ULONG size = sizeof(*value);

    key.fmtid = pci_property_guid;
    return CM_Get_DevNode_PropertyW(devinst, &key, &type, (PBYTE)value, &size, 0) == CR_SUCCESS &&
           type == DEVPROP_TYPE_UINT32 && size == sizeof(*value);
}

static DEVINST nvme_controller(DEVINST devinst) {
    for (int depth = 0; depth < 16; depth++) {
        DWORD base_class;
        DWORD sub_class;
        DWORD prog_if;
        DEVINST parent;

        if (pci_property(devinst, 3, &base_class) && base_class == 0x01 &&
            pci_property(devinst, 4, &sub_class) && sub_class == 0x08 &&
            pci_property(devinst, 5, &prog_if) && prog_if == 0x02) {
            return devinst;
        }
        if (CM_Get_Parent(&parent, devinst, 0) != CR_SUCCESS) {
            break;
        }
        devinst = parent;
    }
    return 0;
}

static int physical_disk_fast(DWORD disk_number) {
    int bus_type = disk_bus_type(disk_number);
    DEVINST devinst;
    DEVINST controller;
    DWORD generation;
    DWORD width;
    DWORD max_generation = 0;
    DWORD max_width = 0;

    if (bus_type != BUS_TYPE_NVME) {
        if (bus_type <= BUS_TYPE_UNKNOWN || bus_type >= BUS_TYPE_MAX ||
            bus_type == BUS_TYPE_USB || bus_type == BUS_TYPE_RAID ||
            bus_type == BUS_TYPE_ISCSI || bus_type == BUS_TYPE_VIRTUAL ||
            bus_type == BUS_TYPE_FILE_BACKED_VIRTUAL || bus_type == BUS_TYPE_SPACES ||
            bus_type == BUS_TYPE_NVMEOF) {
            return AIMDO_DISK_UNKNOWN;
        }
        return AIMDO_DISK_SLOW;
    }
    devinst = disk_devinst(disk_number);
    controller = devinst ? nvme_controller(devinst) : 0;
    if (!controller || !pci_property(controller, 9, &generation) ||
        !pci_property(controller, 10, &width)) {
        return AIMDO_DISK_UNKNOWN;
    }
    pci_property(controller, 11, &max_generation);
    pci_property(controller, 12, &max_width);
    log(DEBUG,
        "%s: disk=%lu current_gen=%lu current_width=%lu max_gen=%lu max_width=%lu\n",
        __func__, disk_number, generation, width, max_generation, max_width);

    return ((generation == 3 && width == 4) ||
            (generation == 4 && width == 4) ||
            (generation == 5 && (width == 2 || width == 4))) ?
           AIMDO_DISK_FAST : AIMDO_DISK_SLOW;
}

SHARED_EXPORT
int aimdo_storage_fast_disk(const wchar_t *path) {
    wchar_t volume[64];
    DWORD *disks = NULL;
    DWORD count = 0;
    int result;
    bool unknown = false;

    if (!path || !get_volume_guid(path, volume) || !get_volume_disks(volume, &disks, &count)) {
        return AIMDO_DISK_UNKNOWN;
    }
    result = cache_lookup(volume, disks, count);
    if (result >= AIMDO_DISK_UNKNOWN) {
        free(disks);
        return result;
    }
    result = AIMDO_DISK_FAST;
    for (DWORD i = 0; i < count; i++) {
        int disk_result = physical_disk_fast(disks[i]);

        if (disk_result == AIMDO_DISK_SLOW) {
            result = AIMDO_DISK_SLOW;
            break;
        }
        if (disk_result == AIMDO_DISK_UNKNOWN) {
            unknown = true;
        }
    }
    if (result == AIMDO_DISK_FAST && unknown) {
        result = AIMDO_DISK_UNKNOWN;
    }
    cache_store(volume, disks, count, result);
    free(disks);
    return result;
}
